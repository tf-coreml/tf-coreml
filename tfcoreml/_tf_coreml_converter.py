from __future__ import print_function
from six import string_types as _string_types

import numpy as np
import tensorflow as tf
from tensorflow.python.util import compat
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes,utils,MLModel
from _ops_to_layers import convert_ops_to_layers
from _interpret_shapes import _interpret_shape as interpret_shape
from _topological_sort import _topological_sort_ops
from optimizations._optimize_nn_spec import optimize_nn_spec

# Context stores useful information about TF graph and the conversion process
class Context(object):
  def __init__(self, consts, shape_dict, ops, blob_graph, output_features):
    self.builder = None
    self.consts = consts
    self.shape_dict = shape_dict #Tensor name --> shape ({str: list})
    self.translated = {x: True for x in self.consts.keys()}
    self.out_name_to_in_name = {} #for blobs which come from a no-op
    self.all_ops = ops
    self.output_names = []
    for out in output_features:
        self.output_names.append(out[0])

    self.skip_map_names = {}
    # Set of all load constants added to the CoreML graph
    self.load_constants_mlmodel = {}

    # Tensor name to list of ops it feeds into
    self.blob_graph = blob_graph
    # Tensor name sto and their inferred rank 4 shape (Batch/Sequennce, C, H, W)
    self.shape_dict_rank_4 = {}
    # Tensor name to labeled shapes (one of 'S','C','H','W').
    # e.g.: 'input' tensor which has shape (1,224,224,3) --> ('S','H','W','C')
    self.dim_labels = {}
    # Whether to use DFS search to infer shapes on the path to conv layers
    self.use_dfs_shape_infer = True #True

def _infer_coreml_input_shape(tf_shape):
  """Infer CoreML input shape from TensorFlow shape.
  """
  if len(tf_shape) == 0:
    shape = [1,1,1]
  elif len(tf_shape) == 1:
    # TODO - remove style transfer 1D hack
    # Input is 1D but it goes to the width dimension: (1,1,W)
    shape = [1, 1, tf_shape[0]]  #(C,H,W)
  elif len(tf_shape) == 2:
    # assume (Batch, Channels) - Batch dimension should be dropped
    shape = [tf_shape[1]]
  elif len(tf_shape) == 3:
    # assume (Batch, Sequence-Length, channels)
    shape = [tf_shape[2], 1, tf_shape[1]]
  elif len(tf_shape) == 4:   #(B,H,W,C) --> (C,H,W)
    shape = [tf_shape[3], tf_shape[1], tf_shape[2]] #(C,H,W)
  else:
    raise ValueError('Unrecognized TensorFlow input shape' + str(tf_shape))
  return shape

def _infer_coreml_output_shape(tf_shape):
  """Infer CoreML output shape from TensorFlow shape.
  """
  shape = []
  if len(tf_shape) == 1:
    shape = [tf_shape[0],1,1]
  elif len(tf_shape) == 2:
    shape = [tf_shape[1]]
  elif len(tf_shape) == 3:
    shape = tf_shape
  elif len(tf_shape) == 4:
    assert tf_shape[0] == 1, "Output 4D tensor's first dimension (Batch) " + \
        "must be 1."
    shape = [tf_shape[3], tf_shape[1], tf_shape[2]] #(C,H,W)
  elif len(tf_shape) == 0: # scalar
    shape = [1]
  else:
    raise ValueError('Unrecognized TensorFlow output shape ' + str(tf_shape))
  return shape

def _convert_pb_to_mlmodel(tf_model_path,
                          mlmodel_path,
                          output_feature_names,
                          input_name_shape_dict = {},
                          image_input_names = None,
                          is_bgr = False,
                          red_bias = 0.0,
                          green_bias = 0.0,
                          blue_bias = 0.0,
                          gray_bias = 0.0,
                          image_scale = 1.0,
                          class_labels = None,
                          predicted_feature_name = None,
                          predicted_probabilities_output = ''):

  # Load the TF graph
  with open(tf_model_path, 'rb') as f:
    serialized = f.read()

  tf.reset_default_graph()
  gdef = tf.GraphDef()
  gdef.ParseFromString(serialized)

  with tf.Graph().as_default() as g:
    tf.import_graph_def(gdef, name='')

  sess = tf.Session(graph=g)
  OPS = g.get_operations()
  OPS = _topological_sort_ops(OPS)

  SHAPE_DICT = dict() #Tensor name --> shape ({str: list})
  CONSTS = dict() #Const Tensor name --> value
  BLOB_GRAPH = {} #Blob name to list of ops it feeds into

  # Make Dictionary of Input blob to the list of ops it feeds into
  for op in OPS:
    for inp in op.inputs:
      if inp.name in BLOB_GRAPH:
        BLOB_GRAPH[inp.name].append(op)
    for out in op.outputs:
      if out.name not in BLOB_GRAPH:
        BLOB_GRAPH[out.name] = []

  # Fill in input information
  input_features = []
  output_features = []
  input_feed_dict = dict() #Input tensors' values

  # run through all placeholders
  for op in OPS:
    output_names = set([compat.as_bytes(x.name) for x in op.outputs])
    if op.type == 'Placeholder':
      # Handle placeholders -- all placeholders are inputs
      assert not filter(output_names.__contains__, output_feature_names), \
          ('Output feature cannot be a placeholder')
      input_name = compat.as_bytes(op.outputs[0].name)
      shape = op.outputs[0].get_shape()
      if not (shape.is_fully_defined() or input_name in input_name_shape_dict):
        assert False,("%s is a placehoder with incomplete shape %s" %(
            input_name, str(shape)))
      if shape.is_fully_defined():
        shape = shape.as_list()
      else:
        shape = input_name_shape_dict[input_name]

      if len(shape) == 0: # scalar - use a 1
        input_feed_dict[op.outputs[0]] = 1
      else:
        input_feed_dict[op.outputs[0]] = np.random.rand(*shape)

      SHAPE_DICT[input_name] = shape

  # Populate SHAPE_DICT:
  # Dictionary for all tensor blobs in the graph and their shapes
  shapes_wanted = []
  for op in OPS:
    for out in op.outputs:
      shape = out.get_shape()
      if not shape.is_fully_defined():
        shapes_wanted.append((compat.as_bytes(out.name), out))
      else:
        SHAPE_DICT[compat.as_bytes(out.name)] = shape.as_list()

  if len(shapes_wanted) > 0:
    print("Shapes not found for %d tensors. "
        "Executing graph to determine shapes. " %(len(shapes_wanted)))
    tensor_names, tensors = zip(*shapes_wanted)
    tensors_evaluated = sess.run(tensors, feed_dict = input_feed_dict)
    for i in range(len(tensor_names)):
      SHAPE_DICT[tensor_names[i]] = list(tensors_evaluated[i].shape)

  # Fill in output information and CONSTS dictionary
  for op in OPS:
    output_names = set([compat.as_bytes(x.name) for x in op.outputs])
    if filter(output_names.__contains__, output_feature_names):
      # retrieve model outputs
      for output in [x for x in op.outputs if x.name in output_feature_names]:
        #infer shape for Core ML
        tf_shape = SHAPE_DICT[compat.as_bytes(output.name)]
        shape = _infer_coreml_output_shape(tf_shape)
        # Objective-C can't handle variable names with colons, replace with __
        # out_name = output.name.replace(':', '__')
        out_name = output.name
        output_features.append((compat.as_bytes(out_name),
            datatypes.Array(*shape)))
    elif op.type == 'Const':
      # retrieve all consts and store them in dictionary
      const = op.outputs[0]
      CONSTS[compat.as_bytes(const.name)] = sess.run(const,
          feed_dict = input_feed_dict)

  assert len(output_features) == len(output_feature_names), \
      'Tensorflow Graph does not contain all the provided Output name(s)'

  #Load all the dictionaries in the object of class context
  context = Context(CONSTS, SHAPE_DICT, OPS, BLOB_GRAPH, output_features)

  # Interpret Input shapes and fill in input information for Core ML
  # (now that SHAPE_DICT and CONSTS are complete)
  for input_tensor in input_feed_dict:
    input_name = compat.as_bytes(input_tensor.name)
    shape = SHAPE_DICT[input_name]

    if context.use_dfs_shape_infer:
      status = interpret_shape(input_name, context)
    else:
      status = False
    if status:
      print('Automatic shape interpretation succeeded for input blob %s' \
          %(input_name))
      shape = context.shape_dict_rank_4[input_name]

    # if the consumer of input_tensor is an one-hot encoding op,
    # treat it as a sequence.
    consumer_op = input_tensor.consumers()[0]
    if consumer_op.type == 'OneHot':
      shape = [1,]
    else:
      shape = _infer_coreml_input_shape(shape)
    input_features.append((compat.as_bytes(input_name),
        datatypes.Array(*shape)))

  # Set classifier flag
  is_classifier = class_labels is not None
  mode = 'classifier' if is_classifier else None

  # Convert the TF graph with builder
  input_features = list(input_features)
  output_features = list(output_features)
  builder = NeuralNetworkBuilder(input_features, output_features, mode=mode)
  context.builder = builder
  convert_ops_to_layers(context)

  # Add image input identifier
  if image_input_names is not None and isinstance(image_input_names,
      _string_types):
    image_input_names = [image_input_names]

  # Add classifier classes (if applicable)
  if is_classifier:
    classes_in = class_labels
    if isinstance(classes_in, _string_types):
      import os
      if not os.path.isfile(classes_in):
        raise ValueError("Path to class labels (%s) does not exist." % \
            classes_in)
      with open(classes_in, 'r') as f:
        classes = f.read()
      classes = classes.splitlines()
    elif type(classes_in) is list: # list[int or str]
      classes = classes_in
    else:
      raise ValueError('Class labels must be a list of integers / strings,'\
          ' or a file path')

    if predicted_feature_name is not None:
      builder.set_class_labels(classes,
          predicted_feature_name = predicted_feature_name,
          prediction_blob = predicted_probabilities_output)
    else:
      builder.set_class_labels(classes)


  # Replace all input/output blob names with ":" to "__" for compatible
  # auto-generated Objective C / Swift code
  interface_blob_names = []
  for idx, in_blob in enumerate(builder.spec.description.input):
    interface_blob_names.append(in_blob.name)
    builder.spec.description.input[idx].name = in_blob.name.replace(':', '__')
  for idx, out_blob in enumerate(builder.spec.description.output):
    interface_blob_names.append(out_blob.name)
    builder.spec.description.output[idx].name = out_blob.name.replace(':', '__')

  nn_spec = builder.nn_spec
  for i, spec_layer in enumerate(nn_spec.layers):
    for j, blob in enumerate(spec_layer.input):
      name = spec_layer.input[j]
      if name in interface_blob_names:
        spec_layer.input[j] = name.replace(':', '__')
    for j, blob in enumerate(spec_layer.output):
      name = spec_layer.output[j]
      if name in interface_blob_names:
        spec_layer.output[j] = name.replace(':', '__')

  if image_input_names is not None:
    for i, img in enumerate(image_input_names):
      image_input_names[i] = img.replace(':', '__')

  # Set pre-processing paramsters
  builder.set_pre_processing_parameters(image_input_names = image_input_names,
                                        is_bgr = is_bgr,
                                        red_bias = red_bias,
                                        green_bias = green_bias,
                                        blue_bias = blue_bias,
                                        gray_bias = gray_bias,
                                        image_scale = image_scale)

  #optimizations on the nn spec
  optimize_nn_spec(builder = builder)

  utils.save_spec(builder.spec, mlmodel_path)
  print("\n Core ML model generated. Saved at location: %s \n" % (mlmodel_path))
  print('Core ML input(s): \n', builder.spec.description.input)
  print('Core ML output(s): \n', builder.spec.description.output)

  # Return the protobuf spec
  spec = builder.spec
  return MLModel(spec)


def convert(tf_model_path,
            mlmodel_path,
            output_feature_names,
            input_name_shape_dict = {},
            image_input_names = None,
            is_bgr = False,
            red_bias = 0.0,
            green_bias = 0.0,
            blue_bias = 0.0,
            gray_bias = 0.0,
            image_scale = 1.0,
            class_labels = None,
            predicted_feature_name = None,
            predicted_probabilities_output = ''):

  """
  Convert a frozen TensorFlow grpah (.pb format) to the CoreML format (.mlmodel)

  Parameters
  ----------
  tf_model_path : str
      Path to the frozen .pb model

  mlmodel_path: str
      Path to where the generated .mlmodel will be stored

  output_feature_names: [str]
      List of strings. Names of the output tensors.

  input_name_shape_dict: {str: [int]}
      Dictionary of input tensor names and their corresponding shapes expressed
      as a list of ints

  Other parameters: Same interpretation as used by Keras/Caffe CoreML
      converters in coremltools

  Returns
  -------
  model: MLModel
      Model in Core ML format.

  """
  return _convert_pb_to_mlmodel(
      tf_model_path,
      mlmodel_path,
      output_feature_names,
      input_name_shape_dict,
      image_input_names = image_input_names,
      is_bgr = is_bgr,
      red_bias = red_bias,
      green_bias = green_bias,
      blue_bias = blue_bias,
      gray_bias = gray_bias,
      image_scale = image_scale,
      class_labels = class_labels,
      predicted_feature_name = predicted_feature_name,
      predicted_probabilities_output = predicted_probabilities_output)

