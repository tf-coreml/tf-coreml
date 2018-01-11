from tensorflow.python.util import compat
import _layers

_OP_REGISTRY = {
    'NoOp': _layers.skip,
    'ExpandDims' : _layers.skip,
    'Cast' : _layers.skip,
    'Squeeze' : _layers.skip,
    'StopGradient' : _layers.skip,
    'CheckNumerics' : _layers.skip,
    'Floor' : _layers.skip, # TODO - need to handle it better
    'Assert' : _layers.skip,
    'Equal' : _layers.skip,
    'All' : _layers.skip,
    'Pack' : _layers.skip, # TODO - need to handle it better
    'SpaceToBatchND':_layers.space_to_batch,
    'BatchToSpaceND':_layers.batch_to_space,
    'ConcatV2' : _layers.concat,
    'GreaterEqual' : _layers.greater, # TODO - need to handle it better
    'LogicalAnd' : _layers.mul, # TODO - need to handle it better
    'BiasAdd' : _layers.add,
    'Slice' : _layers.slice,
    'StridedSlice' : _layers.strided_slice,
    'Fill': _layers.fill,
    'ExtractImagePatches': _layers.extract_image_patches,
    'ArgMax': _layers.argmax,
    # TODO - CoreML not supporting random numbers
    'RandomUniform': _layers.random,
    'Shape': _layers.shape,
    'Maximum': _layers.maximum,
    'RealDiv': _layers.real_div,
    'Transpose': _layers.transpose, # TODO - only works 4D tensors
    'Sigmoid': _layers.sigmoid,
    'ResizeNearestNeighbor': _layers.resize_nearest_neighbor,
    'ResizeBilinear': _layers.resize_bilinear,
    'Square': _layers.square,
    'SquaredDifference': _layers.squared_difference,
    'Pad' : _layers.pad,
    'MirrorPad': _layers.mirror_pad,
    'Mean': _layers.mean, # TODO - there're unsupported configurations
    'Prod': _layers.product, # TODO - there're unsupported configurations
    'Sum': _layers.reduce_sum, # TODO - there're unsupported configurations
    'Max': _layers.reduce_max, # TODO - there're unsupported configurations
    'Min': _layers.reduce_min, # TODO - there're unsupported configurations
    'Greater': _layers.greater, # TODO - only works for x > c where c is const
    'Const': _layers.constant,
    'Softmax': _layers.softmax,
    'Relu6': _layers.relu6,
    'Relu': _layers.relu,
    'QuantizedRelu': _layers.relu,
    'Rsqrt': _layers.rsqrt,
    'Add': _layers.add,
    'Sub': _layers.sub,
    'Mul': _layers.mul,
    'Neg': _layers.neg,
    'MatMul': _layers.inner_product,
    'DepthwiseConv2dNative': _layers.depthwise_conv2d,
    'MaxPool': _layers.maxpool,
    'AvgPool': _layers.avgpool,
    'Conv2DBackpropInput': _layers.deconv2d,
    'Conv2D': _layers.conv2d,
    'QuantizedConv2D': _layers.conv2d,
    'Reshape': _layers.reshape,
    'Concat': _layers.concat,
    'BatchNormWithGlobalNormalization': _layers.batchnorm,
    'Identity': _layers.identity,
    'OneHot': _layers.one_hot,
    'Placeholder': _layers.placeholder,
    'Elu': _layers.elu,
    'QuantizeV2': _layers.skip_one_to_one,
    'QuantizedReshape': _layers.reshape,
    'Dequantize': _layers.skip,
    'RequantizationRange': _layers.skip,
    'Requantize': _layers.skip,
    'Gather': _layers.gather,  # TODO- handled in a very limited setting
    'Reciprocal': _layers.reciprocal,
    'FusedBatchNorm':_layers.batchnorm,
    'LRN': _layers.lrn,
    'Tanh': _layers.tanh,
    'PlaceholderWithDefault': _layers.skip,
    'Log': _layers.log,
    'Minimum': _layers.minimum,
    'Exp': _layers.exp,
    'FloorMod': _layers.floormod #TODO-works when this op's output does not depend on network's input values
}

def _get_translator_function(op_type):
  """Get the right translator function
  """
  if op_type in _OP_REGISTRY:
    return _OP_REGISTRY[op_type]
  else:
    raise TypeError("Translation function missing for op of type %s." % type(op_type))

def connect_skipped_ops(context):
  nn_spec = context.builder.nn_spec
  for layer in nn_spec.layers:
    for i, inp_name in enumerate(layer.input):
      if inp_name in context.skip_map_names:
        layer.input[i] = context.skip_map_names[inp_name]

def check(op, context):
  for inp in op.inputs:
    inp_name = compat.as_bytes(inp.name)
    assert inp_name in context.translated, (
        'No translation found for {}'.format(inp_name))
  for out in op.outputs:
    assert out.name in context.shape_dict, (
        'Shape for {} is not fully defined'.format(out.name))

def translation_required(op, context):
  for out in op.outputs:
    out_name = compat.as_bytes(out.name)
    if out_name in context.translated:
      continue
    else:
      return True
  return False

def stop_translation(context):
  """ Check whether all outputs all translated. If yes return True,
  Otherwise return False
  """
  for out in context.output_names:
    if out not in context.translated:
      return False
  return True

def convert_ops_to_layers(context):
  for i, op in enumerate(context.all_ops):
    if stop_translation(context):
      connect_skipped_ops(context)
      return
    else:
      check(op, context)
      if op.type not in _OP_REGISTRY:
        raise TypeError("Translation function missing for op of type %s." % op.type)
      translator = _get_translator_function(op.type)
      if translation_required(op, context):
        print('%d/%d: Converting op name: %s ( type:  %s )' % (
            i+1, len(context.all_ops), op.name, op.type))
        translator(op, context)
      connect_skipped_ops(context)
