from tensorflow.python.util import compat
import numpy as np
from coremltools.models import *
import ipdb
import _layers

_OP_REGISTERY = {
    'NoOp': _layers.skip,
    'ExpandDims' : _layers.skip,
    'Cast' : _layers.skip,
    'Squeeze' : _layers.skip,
    'StopGradient' : _layers.skip,
    'CheckNumerics' : _layers.skip,
    'Floor' : _layers.skip, #HACK: need to handle it
    'Assert' : _layers.skip,
    'Equal' : _layers.skip,
    'All' : _layers.skip,
    'Pack' : _layers.skip, #HACK: need to handle it
    'SpaceToBatchND':_layers.skip,
    'BatchToSpaceND':_layers.skip,
    'ConcatV2' : _layers.concat,
    'GreaterEqual' : _layers.greater, #HACK
    'LogicalAnd' : _layers.mul, #HACK
    'BiasAdd' : _layers.add,
    'Slice' : _layers.slice,
    'StridedSlice' : _layers.strided_slice,
    'Fill': _layers.fill,
    'ExtractImagePatches': _layers.extract_image_patches,
    'ArgMax': _layers.argmax,
    'RandomUniform': _layers.random, #HACKY
    'Shape': _layers.shape,
    'Maximum': _layers.maximum,
    'RealDiv': _layers.real_div,
    'Transpose': _layers.transpose, # hacky: only works 4D tensors
    'Sigmoid': _layers.sigmoid,
    'ResizeNearestNeighbor': _layers.resize_nearest_neighbor,
    'Square': _layers.square,
    'SquaredDifference': _layers.squared_difference,
    'Pad' : _layers.pad, #why is there no attribute 'mode' in resnet .pb graph's 'Pad' op ?
    'MirrorPad': _layers.mirror_pad,
    'Mean': _layers.mean, #HACKY/INCOMPLETE
    'Prod': _layers.product, #HACKY/INCOMPLETE
    'Sum': _layers.sum, #HACKY/INCOMPLETE
    'Greater': _layers.greater, # hacky: only works for x > c where c is const
    'Const': _layers.constant,
    'Softmax': _layers.softmax,
    'Relu6': _layers.relu6,
    'Relu': _layers.relu,
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
    'Reshape': _layers.reshape, #HACKY/INCOMPLETE
    'Concat': _layers.concat,
    'BatchNormWithGlobalNormalization': _layers.batchnorm,
    'Identity': _layers.identity,
    'Placeholder': _layers.placeholder
}

def _get_translator_function(op_type):
    """Get the right translator function
    """
    if op_type in _OP_REGISTERY:
        return _OP_REGISTERY[op_type]
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
    assert inp_name in context.translated, ('No translation found for {}' \
        .format(inp_name))
  for out in op.outputs:
    assert out.name in context.shape_dict,('Shape for {} is not fully defined'\
        .format(out.name))

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
      if op.type not in _OP_REGISTERY:
        raise TypeError("Translation function missing for op of type %s." % op.type)
      translator = _get_translator_function(op.type)
      if translation_required(op, context):
        print('%d/%d: Converting op name: %s ( type:  %s )' %(i+1, 
            len(context.all_ops), op.name, op.type))
        translator(op, context)
      connect_skipped_ops(context)

        