from tensorflow.python.util import compat

def identity(op, context):
  is_network_output = False
  for out in op.outputs:
    if out.name in context.output_names:
      is_network_output = True
      break
  input_name = compat.as_str_any(op.inputs[0].name)
  for out in op.outputs:
    output_name = compat.as_str_any(out.name)
    if op.inputs[0].op.type != 'Const':
      if is_network_output:
        context.builder.add_activation(
            output_name, 'LINEAR', input_name, output_name, [1.0, 0])
      else:
        skip(op, context)
    context.translated[output_name] = True


def add_const(context, name, x, output_name, shape=None):
  # This is a circular import so we inline the import to avoid this
  from ._shape_sensitive_layers import _add_const
  _add_const(context, name, x, output_name, shape)


def make_tensor(x, context):
  # returns tensor name, after converting input to a tensor, if the input is a
  # const or const-->identity
  if x.op.type == 'Const':
    add_const(context, x.name, context.consts[x.name], x.name)
  elif x.op.type == 'Identity' and x.op.inputs[0].name in context.consts:
    add_const(context, x.name, context.consts[x.op.inputs[0].name], x.name)
  return x.name

#just connect input names to output and record the mapping
def skip(op, context):
  #check if output is one of the network outputs
  # if it is then instead of skip, use an identity layer
  for out in op.outputs:
    if out.name in context.output_names:
      identity(op, context)
      return
  input_names = []
  for inp in op.inputs:
    input_names.append(inp.name)

  if len(input_names) > 1:
    del input_names[1:]

  assert len(input_names) == 1, (
      'Skip op must have only 1 input:' +
      ' This op of type %s cannot be skipped' % (op.type))
  inp_name = input_names[0]
  for out in op.outputs:
    if inp_name not in context.skip_map_names:
      context.skip_map_names[out.name] = inp_name
    else:
      context.skip_map_names[out.name] = context.skip_map_names[inp_name]
    context.translated[out.name] = True
