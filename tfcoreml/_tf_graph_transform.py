
def _create_graph(ops):
  '''
  Creates an adjacency representation of the directed graph formed from the
  list of TF ops.
  Nodes are the ops. Directed edge from an op "A" to an op "B" implies that at
  least one of the outputs of A is feeding as an input to B.

  :param: list of ops. List of size N.
  :return: G: list of lists. Outer list is of size N.

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


def _find_skippable_ops(G, ops, output_names):
  '''
  :param G: graph as described in _create_graph 
  :param ops: list of TF graph ops
  :param output_names: [str]: list of output names
  :return: a set of op names that can be skipped during conversion
           because they do not connect to the output
  '''

  #first reverse the graph
  n = len(ops)
  reverse_G = [[] for i in range(n)]
  for i, child_list in enumerate(G):
    for j in child_list:
      reverse_G[j].append(i)

  #ids of all unvisited ops: initially all the ops are unvisited
  unvisited_op_ids = set(range(n))

  #get ids of ops that produce the network output nodes:
  #these will be the start nodes for our graph traversal
  start_ids = []
  for i, op in enumerate(ops):
    for out in op.outputs:
      if out.name in output_names:
        start_ids.append(i)

  if len(start_ids) == 0:
    raise ValueError('No op found in the TF graph that produces the given output name(s)')

  #Lets do BFS Graph traversal
  #(on the reverse TF graph starting from output producing ops)
  from collections import deque
  list_queue = deque()
  for idx in start_ids:
    #Mark idx as visited and put idx in queue
    if idx in unvisited_op_ids:
      unvisited_op_ids.remove(idx)
      list_queue.append(idx)

    while len(list_queue) > 0:
      op_id = list_queue.popleft()
      for child_op in reverse_G[op_id]:
        if child_op in unvisited_op_ids:
          unvisited_op_ids.remove(child_op)
          list_queue.append(child_op)

  #Collect all unvisited ops
  skip_ops = set()
  for i in unvisited_op_ids:
    skip_ops.add(ops[i].name)
  return skip_ops

def _topological_sort_ops(ops, output_names):
  '''
  :param ops: list of TF ops
  :param output_names: [str]: list of output names
  :return: list of TF ops, in topological sort order such that an op is
  encountered only after all the ops that generated its inputs have been
  visited.
  And also return a set of op names that can be skipped during conversion,
  as they are not connected to the output

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
    node = list(not_visited.keys())[0]
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

  skip_ops = _find_skippable_ops(G, ops, output_names)

  return [x for _, x in sorted(zip(topological_label, ops))], skip_ops
