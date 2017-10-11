import _optimize 
from coremltools.proto import NeuralNetwork_pb2 as _NeuralNetwork_pb2
import ipdb

def _optimize_fold_load_constants(nn_layers):
  """
  Fold load constants that interact through 'add', 'multiply', 'activation' or 'unary' layers. 
  In other words, evaluate any sub-graph that involves only 'load_constant', 'multiply', 'add', 'activation'
  or 'unary' layer types and replace it with a single load constant layer. 
  """
  
  _optimize._fold_constants(nn_layers)
  
def _optimize_conv_mul_add(nn_layers):
  """
  Detect Multiply or add layers after convolution and recast as Batchnorm layer so that it can be fused in the framework. 
  """  
  _optimize._fuse_conv_mul_add(nn_layers)
  
  
def _optimize_spatial_reduce_operation(nn_layers):
  
  """
  Find a reduce layer with mode = 'average'/'max' and axis = 'HW'
  and replace it with global average/max pooling layer.
  """
  
  _optimize._spatial_reduce_as_global_pool(nn_layers)
    
def optimize_nn_spec(nn_spec):
  
  """
  Call a specific set of network optimizations
  """
  
  _optimize_fold_load_constants(nn_spec.layers)  
  _optimize_spatial_reduce_operation(nn_spec.layers)
  _optimize_conv_mul_add(nn_spec.layers)
  
  
  