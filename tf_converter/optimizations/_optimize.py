import numpy as np
from coremltools.proto import NeuralNetwork_pb2 as _NeuralNetwork_pb2

def _evaluate_unary(layer, x):
  params = layer.unary
  x = x * params.scale + params.shift
  op_type = _NeuralNetwork_pb2.UnaryFunctionLayerParams.Operation.Name(params.type)
  if op_type == 'SQRT':
    return np.sqrt(x)
  elif op_type == 'RSQRT':
    return 1/np.sqrt(x + params.epsilon)
  elif op_type == 'INVERSE':
    return 1/(x + params.epsilon)
  elif op_type == 'POWER':
    return x ** params.alpha
  elif op_type == 'EXP':
    return np.exp(x)
  elif op_type == 'LOG':
    return np.log(x)
  elif op_type == 'ABS':
    return np.abs(x)
  elif op_type == 'THRESHOLD':
    return np.maximum(x, params.alpha)  
  else:
    raise ValueError('Unary function operation type not recognized: %s' %(op_type))    
    
    
def _evaluate_activaton(layer, x, shape):
  params = layer.activation
  act_type = params.WhichOneof('NonlinearityType')
  if act_type == 'linear':
    return params.linear.alpha*x + params.linear.beta
  elif act_type == 'ReLU':
    return np.maximum(0,x)
  elif act_type == 'leakyReLU':
    return (x<0)*params.leakyReLU.alpha*x + (x>=0)*x
  elif act_type == 'thresholdedReLU':
    return x*(x>params.thresholdedReLU.alpha)
  elif act_type == 'PReLU':
    alpha = np.reshape(params.PReLU.alpha,(1, shape[0],1,1))
    x = np.reshape(x, shape)
    alpha = np.broadcast_to(alpha, shape)
    return np.maximum(x,0) + alpha * np.minimum(x,0)    
  elif act_type == 'tanh':
    return np.tanh(x)
  elif act_type == 'scaledTanh':
    return params.scaledTanh.alpha * np.tanh(x * params.scaledTanh.beta)
  elif act_type == 'sigmoid':
    return 1. / (1 + np.exp(-x))   
  elif act_type == 'sigmoidHard':
    return np.minimum(np.maximum((params.sigmoidHard.alpha * x) + params.sigmoidHard.beta, 0), 1)  
  elif act_type == 'ELU':
    return x*(x>=0) + params.ELU.alpha*(np.exp(x)-1)*(x<0)  
  elif act_type == 'softsign':
    return x/(np.abs(x)+1)
  elif act_type == 'softplus':
    return np.log(1 + np.exp(x))
  elif act_type == 'parametricSoftplus':    
    alpha = np.reshape(alpha,(1, shape[0],1,1))
    alpha = np.broadcast_to(alpha, shape)
    beta = np.reshape(beta,(1, shape[0],1,1))
    beta = np.broadcast_to(beta, shape)
    x = np.reshape(x, shape)
    return params.parametricSoftplus.alpha*np.log(1 + np.exp(params.parametricSoftplus.beta * x))
  else:
    raise ValueError('Activation type not recognized: %s' %(act_type))   
      
def _replace_with_load_constant(nn_layers, ind, data, shape, load_constant_outputs):
  nn_layers[ind].ClearField("input")
  nn_layers[ind].loadConstant.MergeFromString('')
  params = nn_layers[ind].loadConstant
  params.data.floatValue.extend(map(float, data.flatten())) 
  params.shape.extend(shape)
  load_constant_outputs[nn_layers[ind].output[0]] = (data.flatten(), shape)   
  
def _spatial_reduce_as_global_pool(nn_layers):
  reduce_layers_replace_pooling = []
  for i, layer in enumerate(nn_layers):
    layer_type = layer.WhichOneof('layer')
    #print('%d/%d: Layer of type %s' %(i+1, len(nn_layers), layer_type))
    if layer_type == 'reduce':
      params = layer.reduce
      axis = _NeuralNetwork_pb2.ReduceLayerParams.ReduceAxis.Name(params.axis)
      if axis == 'HW':
        mode = _NeuralNetwork_pb2.ReduceLayerParams.ReduceOperation.Name(params.mode)
        if mode == 'AVG':
          reduce_layers_replace_pooling.append((i, 'AVERAGE'))
        if mode == 'MAX':
          reduce_layers_replace_pooling.append((i, 'MAX'))  
  
  for replace in reduce_layers_replace_pooling: 
    nn_layers[replace[0]].pooling.MergeFromString('')
    params = nn_layers[replace[0]].pooling
    params.type = _NeuralNetwork_pb2.PoolingLayerParams.PoolingType.Value(replace[1])
    params.globalPooling = True
    params.valid.MergeFromString('')  
    
    
def _remove_disconnected_load_constants(nn_layers):
  load_constant_outputs = dict()
  for i, layer in enumerate(nn_layers):
    layer_type = layer.WhichOneof('layer')
    if layer_type == 'loadConstant': load_constant_outputs[layer.output[0]] = i
    
    for inp in layer.input:
      if inp in load_constant_outputs:
        load_constant_outputs.pop(inp)
      
  for index in sorted(load_constant_outputs.values(), reverse=True):
    del nn_layers[index]
    
                      
def _fold_constants(nn_layers):
  load_constant_outputs = {}
  for i, layer in enumerate(nn_layers):
    layer_type = layer.WhichOneof('layer')

    if layer_type == 'loadConstant':
      load_constant_outputs[layer.output[0]] = (np.array(layer.loadConstant.data.floatValue), np.array(layer.loadConstant.shape).astype(np.int))
       

    if layer_type == 'unary' and layer.input[0] in load_constant_outputs:
      x = load_constant_outputs[layer.input[0]][0]
      shape = load_constant_outputs[layer.input[0]][1]
      y = _evaluate_unary(layer, x)
      _replace_with_load_constant(nn_layers, i, y, shape, load_constant_outputs)
      
    if layer_type == 'activation' and layer.input[0] in load_constant_outputs:
      x = load_constant_outputs[layer.input[0]][0]
      shape = load_constant_outputs[layer.input[0]][1]
      y = _evaluate_activaton(layer, x, shape)
      _replace_with_load_constant(nn_layers, i, y, shape, load_constant_outputs)  
      
    if layer_type == 'multiply' or layer_type == 'add':
      all_constant_inputs = True
      for inp in layer.input:
        if inp not in load_constant_outputs:
          all_constant_inputs = False
          break   
          
      if all_constant_inputs:
        x = load_constant_outputs[layer.input[0]][0]
        shape = load_constant_outputs[layer.input[0]][1]
        if len(layer.input) == 1:
          x = x + layer.alpha if layer_type == 'add' else x * layer.alpha
        else:
          for j, inp in enumerate(layer.input):
            if j == 0: continue
            shape = np.maximum(shape, load_constant_outputs[inp][1])        
            xj = load_constant_outputs[inp][0]
            x = x + xj if layer_type == 'add' else x * xj
        _replace_with_load_constant(nn_layers, i, x, shape, load_constant_outputs)
      
  _remove_disconnected_load_constants(nn_layers)        
        
def _fuse_conv_mul_add(nn_layers):
  #first create 2 dictionaries
  blob_dst = dict() #blob name to the list of indices of the layers it feeds into
  blob_src = dict() #blob name to the index of the layer it is coming from
  for i, layer in enumerate(nn_layers):
    for inp in layer.input:
      if inp in blob_dst:
        blob_dst[inp].append(i)
      else:
        blob_dst[inp] = [i]
    for out in layer.output:
      if out in blob_src:
        raise ValueError('Blob %s has been generated by more than 1 layers' %(out))
      blob_src[out] = i
      
  def is_followed_by_muladd_constant(out):
    if out in blob_dst and len(blob_dst[out]) == 1:
      next_layer_id = blob_dst[out][0]
      next_layer = nn_layers[next_layer_id]
      if next_layer.WhichOneof('layer') == 'multiply' or next_layer.WhichOneof('layer') == 'add':
        if len(next_layer.input) == 2:
          other_input = next_layer.input[1] if next_layer.input[0] == out else next_layer.input[0]
          other_input_src_layer = nn_layers[blob_src[other_input]]
          if other_input_src_layer.WhichOneof('layer') == 'loadConstant':
            _,H,W = other_input_src_layer.loadConstant.shape
            if H==1 and W==1:
              x = np.array(other_input_src_layer.loadConstant.data.floatValue)
              return True, x, next_layer_id, next_layer.output[0]
    return False, None, None, None
    
  def cast_two_layers_as_bn(x1, x2, conv_out, id1, id2):
    layer1_type = nn_layers[id1].WhichOneof('layer')
    layer2_type = nn_layers[id2].WhichOneof('layer')
    #convert the second layer into batchnorm
    nn_layers[id2].batchnorm.MergeFromString('')
    params = nn_layers[id2].batchnorm
    nn_layers[id2].ClearField("input")
    nn_layers[id2].input.append(conv_out)
    C = len(x1)
    params.channels = C
    gamma = np.ones((C))
    beta = np.zeros((C))
    variance = np.ones((C))
    mean = np.zeros((C))
    if layer1_type == 'add' and layer2_type == 'multiply':
      gamma = x2
      beta = x1 * x2
    if layer1_type == 'add' and layer2_type == 'add':
      beta = x1 + x2
    if layer1_type == 'multiply' and layer2_type == 'multiply':
      gamma = x1 * x2
    if layer1_type == 'multiply' and layer2_type == 'add':
      gamma = x1
      beta = x2     
    params.gamma.floatValue.extend(map(float, gamma.flatten()))
    params.beta.floatValue.extend(map(float, beta.flatten()))
    params.mean.floatValue.extend(map(float, mean.flatten()))
    params.variance.floatValue.extend(map(float, variance.flatten()))  
    
  def cast_one_layer_as_bn(x, conv_out, id):
    layer_type = nn_layers[id].WhichOneof('layer')
    #convert the layer into batachnorm layer
    nn_layers[id].batchnorm.MergeFromString('')
    params = nn_layers[id].batchnorm
    nn_layers[id].ClearField("input")
    nn_layers[id].input.append(conv_out)
    C = len(x)
    params.channels = C
    gamma = np.ones((C))
    beta = np.zeros((C))
    variance = np.ones((C))
    mean = np.zeros((C))
    if layer_type == 'add':
      beta = x
    if layer_type == 'multiply':
      gamma = x   
    params.gamma.floatValue.extend(map(float, gamma.flatten()))
    params.beta.floatValue.extend(map(float, beta.flatten()))
    params.mean.floatValue.extend(map(float, mean.flatten()))
    params.variance.floatValue.extend(map(float, variance.flatten()))
          
        
  layers_to_be_removed = []    
  #now go through the layers and look for "conv + mul/add" or "conv + mul/add + add/mul" patterns
  for i, layer in enumerate(nn_layers):
    layer_type = layer.WhichOneof('layer')
     
    #the pattern matching can go very deep
    if layer_type == 'convolution' and layer.convolution.isDeconvolution == False:
      conv_out = layer.output[0]
      #check if its followed by a 'multiply' or 'add'
      status_1, x_1, layer_id_1, layer_1_out = is_followed_by_muladd_constant(conv_out)
      if status_1:
        status_2, x_2, layer_id_2, _ = is_followed_by_muladd_constant(layer_1_out)
        if status_2:
          if len(x_1) == len(x_2):
            cast_two_layers_as_bn(x_1, x_2, conv_out, layer_id_1, layer_id_2)
            layers_to_be_removed.append(layer_id_1)
        else:
          cast_one_layer_as_bn(x_1, conv_out, layer_id_1)   

  for index in sorted(layers_to_be_removed, reverse=True):
    del nn_layers[index]
  _remove_disconnected_load_constants(nn_layers)   
                 
                 
                
           
           
               
       
           
    
             
  
  
  