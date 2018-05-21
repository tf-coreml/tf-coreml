from tensorflow.python.util import compat
from coremltools.proto import NeuralNetwork_pb2

def identity(op, context, input_name = None):
  is_network_output = False
  for out in op.outputs:
    if out.name in context.output_names:
      is_network_output = True
      break
  if input_name is None:
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
def skip(op, context, input_name = None):
  #check if output is one of the network outputs
  # if it is then instead of skip, use an identity layer
  for out in op.outputs:
    if out.name in context.output_names:
      identity(op, context, input_name)
      return

  input_names = []

  if input_name is not None:
    input_names.append(input_name)
  else:
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

def effectively_constant_op(op, context):
  for out in op.outputs:
    x = context.session.run(out, feed_dict=context.input_feed_dict)
    add_const(context, out.name, x, out.name)
    context.translated[out.name] = True

def custom_layer(op, context):

  print("Adding custom layer")

  if op.name in context.custom_conversion_functions or \
     op.type in context.custom_conversion_functions:

    if op.type in context.custom_conversion_functions:
      func = context.custom_conversion_functions[op.type]
    else:
      func = context.custom_conversion_functions[op.name]

    # Fill up values of any constant inputs that this op receives
    constant_inputs = {}
    for inp_ in op.inputs:
      if inp_.name in context.consts:
        constant_inputs[inp_.name] = context.consts[inp_.name]
      elif inp_.op.type == 'Identity' and inp_.op.inputs[0].name in context.consts:
        constant_inputs[inp_.op.inputs[0].name] = context.consts[inp_.op.inputs[0].name]

    kwargs = {"op": op, "nn_builder": context.builder, "context": context, "constant_inputs": constant_inputs}
    func(**kwargs)

  else:
    params = NeuralNetwork_pb2.CustomLayerParams()
    params.className = op.type
    params.description = "Custom layer that corresponds to the TensorFlow op {}".format(op.type, )
    inputs = [inp.name for inp in op.inputs]
    outputs = [out.name for out in op.outputs]
    context.builder.add_custom(name=op.name,
                            input_names=inputs,
                            output_names=outputs,
                            custom_proto_spec=params)
  for out in op.outputs:
    context.translated[out.name] = True
  context.ops_converted_to_custom_layers.append(op)

