
def _create_graph(ops):
  '''
  Creates an adjacency representation of the directed graph formed from the
  list of TF ops.
  Nodes are the ops. Directed edge from an op "A" to an op "B" implies that at
  least one of the outputs of A is feeding as an input to B.

  input: list of ops. List of size N.
  output: G: list of lists. Outer list is of size N.

  Let the number of ops be N.
  Then the adjacency representation is a list of lists: G. G[i] is the list of
  all ops that have directed edges impingent from op "i",
  i.e. G[i] is the fan-out list of op "i"
  '''
  n = len(ops)
  G = [[] for i in range(n)]
  op_name_to_index = dict()
  #First pass to assign op name to its index
  for i, op in enumerate(ops):
    op_name_to_index[op.name] = i
  #Second pass to construct the graph
  for i, op in enumerate(ops):
    for inp in op.inputs:
      G[op_name_to_index[inp.op.name]].append(i)

  return G

def _push_stack(stack, node, in_stack):
  stack.append(node)
  if node in in_stack:
    raise ValueError('Graph has cycles.')
  else:
    in_stack[node] = True

def _get_unvisited_child(G, node, not_visited):
  for child in G[node]:
    if child in not_visited:
      return child
  return -1

def _topological_sort_ops(ops):
  '''
  input: list of TF ops
  output: list of TF ops, in topological sort order such that an op is
  encountered only after all the ops that generated its inputs have been
  visited.

  As a by product, also checks if the graph has cycles. Raises an error if
  it does.
  '''

  G = _create_graph(ops)
  n = len(ops)
  # Topological label for each op. Highest will be for the sink nodes.
  topological_label = [-1 for i in range(n)]
  stack = []
  in_stack = dict()
  not_visited = dict.fromkeys([i for i in range(n)])
  label_counter = n-1

  while len(not_visited) > 0:
    node = not_visited.keys()[0]
    _push_stack(stack, node, in_stack)
    while len(stack) > 0:
      node = _get_unvisited_child(G, stack[-1], not_visited)
      if node != -1:
        _push_stack(stack, node, in_stack)
      else:
        node = stack.pop()
        in_stack.pop(node)
        not_visited.pop(node)
        topological_label[node] = label_counter
        label_counter -= 1

  return [x for _, x in sorted(zip(topological_label, ops))]
