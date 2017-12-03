from tensorflow.python.util import compat
import numpy as np
from _interpret_shapes import _interpret_shape as interpret_shape
import _layers

def _remove_beginning_unit_dimensions(in_tuple):
  for i, value in enumerate(in_tuple):
    if value == 1:
      continue
    else:
      return in_tuple[i:]

def _add_const(context, name, x, output_name, shape=None):
  if output_name in context.load_constants_mlmodel:
    return

  if shape is not None:
    context.builder.add_load_constant(name, output_name, x, shape)
    context.load_constants_mlmodel[output_name] = True
    return

  context.load_constants_mlmodel[output_name] = True

  if context.use_dfs_shape_infer:
    status = interpret_shape(output_name, context)
  else:
    status = False

  if status:
    rank_4_shape = context.shape_dict_rank_4[output_name]
    # TODO - Interpreting 1st dimension as seq. in this case instead of batch
    seq, h, w, c = rank_4_shape
    x = np.reshape(x, (seq, h, w, c))
    #first check the simple case: seq. dimension is 1
    if seq == 1:
      shape = [c, h, w] # (C, H, W)
      x = np.transpose(x, [0, 3, 1, 2])
      context.builder.add_load_constant(name, output_name, x, shape)

    #when sequence dimension is not 1, we need some permute layers as well
    #since CoreML only allows loading constant of rank-3: [C,H,W])
    else:
      assert c == 1 or h == 1 or w == 1, \
          'Add constant: cannot add a constant in which all the dimensions ' \
          '(Seq, C, H, W) are of non-unit size'
      if c == 1: #swap seq. and C
        x = np.transpose(x, [3, 0, 1, 2]) #(S,H,W,C) --> (C,S,H,W)
        context.builder.add_load_constant(
            name + '_pre_permute', output_name + '_pre_permute', x, [seq, h, w])
        context.builder.add_permute(
            output_name, (1, 0, 2, 3), output_name + '_pre_permute', output_name)
      elif h == 1: #swap seq. and H
        x = np.transpose(x, [1, 3, 0, 2]) #(S,H,W,C) --> (H,C,S,W)
        context.builder.add_load_constant(
            name + '_pre_permute', output_name + '_pre_permute', x, [c, seq, w])
        context.builder.add_permute(
            output_name, (2, 1, 0, 3), output_name + '_pre_permute', output_name)
      else: # w == 1, swap seq. and W
        x = np.transpose(x, [2, 3, 1, 0]) #(S,H,W,C) --> (W,C,H,S)
        context.builder.add_load_constant(
            name + '_pre_permute', output_name + '_pre_permute', x, [c, h, seq])
        context.builder.add_permute(
            output_name, (3, 1, 2, 0), output_name + '_pre_permute', output_name)

  else: #Static shape mapping
    shape = list(x.shape)
    assert len(shape) < 5, 'Const blob shape is more than rank 4'
    if len(shape) == 0:
      shape = [1, 1, 1]      #(1,1,1)
    elif len(shape) == 1:
      shape = [shape[0], 1, 1] #(C,1,1)
    elif len(shape) == 2:
      shape = [shape[1], 1, shape[0]] # HACK: (W,C) ---> (C,1,W) . Style transfer matrices are (W,C)
      x = np.transpose(x, [1, 0])
    elif len(shape) == 3:
      shape = [shape[2], shape[0], shape[1]] # (H,W,C) ---> (C,H,W)
      x = np.transpose(x, [2, 0, 1])
    elif len(shape) == 4:
      assert shape[0] == 1, 'Add Constant: Batch dimension must be 1'
      shape = [shape[3], shape[1], shape[2]]  #(B,H,W,C) ---> (C,H,W)
      x = x[0, :, :, :] #(H,W,C)
      x = np.transpose(x, [2, 0, 1])
    context.builder.add_load_constant(name, output_name, x, shape)


def _add_concat(op, context):

  output_name = compat.as_bytes(op.outputs[0].name)
  output_shape = context.shape_dict[output_name]
  axis = 3 #3 -> 'Channel', 2 -> 'Width', 1 -> 'Height

  if op.type == 'Concat':
    axis_name = compat.as_bytes(op.inputs[0].name)
    axis = context.consts[axis_name]
    input_names = []
    for i, input in enumerate(op.inputs):
      if i == 0:
        continue
      input_names.append(compat.as_bytes(input.name))

  if op.type == 'ConcatV2':
    axis_name = compat.as_bytes(op.inputs[-1].name)
    axis = context.consts[axis_name]
    input_names = []
    for i, input in enumerate(op.inputs):
      if i == len(op.inputs) - 1:
        continue
      input_names.append(compat.as_bytes(input.name))

  if context.use_dfs_shape_infer:
    status = interpret_shape(output_name, context)
  else:
    status = False

  if status:
    labeled_shape = context.dim_labels[output_name]
    if labeled_shape[axis] == 'C':
      axis = 3
    elif labeled_shape[axis] == 'H':
      axis = 1
    elif labeled_shape[axis] == 'W':
      axis = 2
    else:
      assert False, 'Concatenation supported only along channel, height or '\
          'width dimensions'
  else:
    if len(output_shape) == 4:
      assert axis in [1, 2, 3], 'Concat axis case not handled'
    elif len(output_shape) == 3:
      axis += 1
    elif len(output_shape) == 1:
      axis = 3
    else:
      assert False, 'Concat axis case not handled'

  # Temporary workaround for fixing bugs on certain devices.
  # TODO: remove this in future
  # If concat's input is coming from another pool/concat: insert a linear activation layer
  coreml_layers = context.builder.nn_spec.layers
  for layer in coreml_layers:
    if layer.WhichOneof('layer') in ['concat', 'pooling']:
      for i, inp in enumerate(input_names):
        if layer.output[0] == inp:
          out = inp + '__linear_activation'
          context.builder.add_activation(out, 'LINEAR', inp, out, [1.0, 0])
          input_names[i] = out

  if axis == 3: #concatenate along channel axis
    context.builder.add_elementwise(
        output_name, input_names, output_name, 'CONCAT')
  elif axis == 2: #concatentae along width axis
    blob_postfix = '_swap_W_C_'
    transpose_order = (0, 3, 2, 1)
    inputs_permuted = []
    for i, input_name in enumerate(input_names):
      context.builder.add_permute(
          output_name + '_' + str(i), transpose_order,
          input_name, input_name + blob_postfix + str(i))
      inputs_permuted.append(input_name + blob_postfix + str(i))
    context.builder.add_elementwise(
        output_name + '_concat', inputs_permuted, output_name + '_concat', 'CONCAT')
    context.builder.add_permute(
        output_name, transpose_order, output_name + '_concat', output_name)
  elif axis == 1: #concatentae along heigth axis
    inputs_permuted = []
    for i, input_name in enumerate(input_names):
      context.builder.add_permute(
          output_name + '_' + str(i), (0, 2, 1, 3),
          input_name, input_name + '_swap_H_C_' + str(i))
      inputs_permuted.append(input_name + '_swap_H_C_' + str(i))
    context.builder.add_elementwise(
        output_name + '_concat', inputs_permuted, output_name + '_concat', 'CONCAT')
    context.builder.add_permute(
        output_name, (0, 2, 1, 3), output_name + '_concat', output_name)
  else:
    assert False, 'Concat axis case not handled'
  context.translated[output_name] = True

def _add_reshape(op, context):
  input_name = compat.as_bytes(op.inputs[0].name)
  output_name = compat.as_bytes(op.outputs[0].name)

  #First make sure the the input blob exists in the CoreML graph
  input_name = _layers.make_tensor(op.inputs[0], context)

  input_shape = context.shape_dict[input_name]
  target_shape = context.shape_dict[output_name]

  squeezed_input_shape = _remove_beginning_unit_dimensions(input_shape)
  squeezed_output_shape = _remove_beginning_unit_dimensions(target_shape)
  if squeezed_input_shape == squeezed_output_shape:
    # reshape is either squeeze or expand_dim
    _layers.skip(op, context)
    return

  if context.use_dfs_shape_infer:
    status = interpret_shape(output_name, context)
  else:
    status = False

  if status:
    target_shape = context.shape_dict_rank_4[output_name]
    if interpret_shape(input_name, context):
      input_shape_rank_4 = context.shape_dict_rank_4[input_name]
      if input_shape_rank_4 == target_shape:
        _layers.skip(op, context)
        return

  # When reshape is immediately followed by squeeze
  if len(op.outputs) > 0 and len(op.outputs[0].consumers()) > 0 and \
      op.outputs[0].consumers()[0].type == 'Squeeze':
    squeezed_output_name = compat.as_bytes(
        op.outputs[0].consumers()[0].outputs[0].name)
    target_shape = context.shape_dict[squeezed_output_name]

  # TODO - these cases of reshape are just for mobilenet and stylenet:
  # if target_shape == (1,X) ----> new_shape = (X,1,1)
  # if targt_shape == (X,1) -----> new_shape = (1,1,X)
  assert len(target_shape) in [1, 2, 3, 4], (
      'Reshape: Currently only supported if target shape is rank 2, 3 or 4')

  mode = 0
  if len(target_shape) == 2:
    if target_shape[1] != 1: #(1,X)
      new_shape = (1, target_shape[1], 1, 1)
      if len(input_shape) == 4 or len(input_shape) == 3:
        # (N,H,W,C) --> (1,C) or (N,S,C) --> (N,1,W,C)
        mode = 1
    else:
      new_shape = (1, 1, 1, target_shape[0])
    context.builder.add_reshape(
        output_name, input_name, output_name, new_shape, mode)
  elif len(target_shape) == 3:
    # Target shape is [H,W,C] --> [1, C, H, W]
    new_shape = (1, target_shape[2], target_shape[0], target_shape[1])
    context.builder.add_reshape(
        output_name, input_name, output_name, new_shape, 1)
  elif len(target_shape) == 4:
    new_shape = (
        target_shape[0], target_shape[3], target_shape[1], target_shape[2])
    context.builder.add_reshape(
        output_name, input_name, output_name, new_shape, 1)
  elif len(target_shape) == 1:
    new_shape = (1,target_shape[0],1,1)
    context.builder.add_reshape(
      output_name, input_name, output_name, new_shape, 1)

  context.translated[output_name] = True

# TODO - sum, max and mean looks all like reduce, clean up once it's correct
def _add_reduce(op, context, mode):

  input_name = compat.as_bytes(op.inputs[0].name)
  output_name = compat.as_bytes(op.outputs[0].name)
  axis_ind = context.consts[op.inputs[1].name]

  input_shape = context.shape_dict[input_name]

  if context.use_dfs_shape_infer:
    status = interpret_shape(input_name, context)
  else:
    status = False

  # Determine reduction axis labels
  axis = None
  if status:
    labeled_shape = context.dim_labels[input_name]
    if isinstance(axis_ind, np.ndarray):
      axis = ''
      for i in axis_ind:
        if input_shape[i] != 1:
          axis += labeled_shape[i]
      axis = ''.join(sorted(axis))
    else:
      axis = labeled_shape[axis_ind]
    assert axis in ['S', 'C', 'H', 'W', 'CHW', 'HW'], (
        'Axis value %s not supported. '
        'Reduction supported along C, H, W, HW, CHW dimensions only.' % axis)
  else:
    if isinstance(axis_ind, np.ndarray):
      axis_ind = axis_ind.tolist()
      if len(axis_ind) == 1:
        axis_ind = axis_ind[0]
      elif len(input_shape) == len(axis_ind):
        axis = 'CHW'
    if axis is None:
      # single axis reduction
      axis_ind = (len(input_shape) + axis_ind) if axis_ind < 0 else axis_ind
      if len(input_shape) == 4:
        if axis_ind == 3:
          axis = 'C'
      elif len(input_shape) == 2:
        if axis_ind == 0:
          # TODO - only works for stylenet. (W,C)--->(1,C)
          axis = 'W'
        elif axis_ind == 1:
          axis = 'C'
      elif len(input_shape) == 1:
        if axis_ind == 0:
          axis = 'CHW'

  if axis == None:
    raise NotImplementedError(
        'Reduce axis %s for input shape %s not handled currently' %(str(axis_ind), str(input_shape)))

  # The simple case; reduction along non sequence axis
  if axis != 'S':
    context.builder.add_reduce(output_name, input_name, output_name, axis, mode)
  # Need to permute, reduce and then permute back
  else:
    context.builder.add_permute(
        output_name, (1, 0, 2, 3), input_name, output_name + '_swap_Seq_C')
    context.builder.add_reduce(
        output_name, output_name + '_swap_Seq_C',
        output_name + '_pre_permute', 'C', mode)
    context.builder.add_permute(
        output_name, (1, 0, 2, 3), output_name + '_pre_permute', output_name)
  context.translated[output_name] = True

def _add_mean(op, context):

  input_name = compat.as_bytes(op.inputs[0].name)
  output_name = compat.as_bytes(op.outputs[0].name)
  axis_ind = context.consts[op.inputs[1].name]

  input_shape = context.shape_dict[input_name]

  if context.use_dfs_shape_infer:
    status = interpret_shape(input_name, context)
  else:
    status = False

  if status:
    labeled_shape = context.dim_labels[input_name]
    if isinstance(axis_ind, np.ndarray):
      axis = ''
      for i in axis_ind:
        if input_shape[i] != 1:
          axis += labeled_shape[i]
      axis = ''.join(sorted(axis))
    else:
      axis = labeled_shape[axis_ind]
    assert axis in ['S', 'C', 'H', 'W', 'CHW', 'HW'], (
        'Axis value %s not supported. '
        'Reduction supported along C, H, W, HW, CHW dimensions only.' % axis)
  else:
    if len(input_shape) == 4 and (
        np.array_equal(axis_ind, np.array([0, 1, 2])) or
        np.array_equal(axis_ind, np.array([1, 2]))):
      axis = 'HW'
    else:
      assert False, 'Mean axis case not handled currently'

  mode = 'avg'
  # The simple case; reduction along non sequence axis
  if axis != 'S':
    context.builder.add_reduce(output_name, input_name, output_name, axis, mode)
  # Need to permute, reduce and then permute back
  else:
    context.builder.add_permute(
        output_name, (1, 0, 2, 3), input_name, output_name + '_swap_Seq_C')
    context.builder.add_reduce(
        output_name, output_name + '_swap_Seq_C',
        output_name + '_pre_permute', 'C', mode)
    context.builder.add_permute(
        output_name, (1, 0, 2, 3), output_name + '_pre_permute', output_name)
  context.translated[output_name] = True
