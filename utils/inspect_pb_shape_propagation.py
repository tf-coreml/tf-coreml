import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import time
import operator

graph_def = graph_pb2.GraphDef()
with open("./style_opt.pb", "rb") as f:
    graph_def.ParseFromString(f.read())

tf.import_graph_def(graph_def)

sess = tf.Session()
OPS = sess.graph.get_operations()

blob_to_optype_output_dict = {}

time.sleep(1)
for i, op in enumerate(OPS):    
    # print '---------------------------------------------------------------------------------------------------------------------------------------------'
    # print i, 'op name=', op.name, 'op type= (', op.type, ') inputs = ', [x.name for x in op.inputs], 'outputs=', [x.name for x in op.outputs]
    # print '@input shapes:'
    # for x in op.inputs:
    #     print 'name= ', x.name, ':', x.get_shape()
    # print '@output shapes:'
    # for x in op.outputs:
    #     print 'name= ' , x.name, ':', x.get_shape()

    n_inputs = len(op.inputs)
    n_outputs = len(op.outputs)
    if n_inputs > 0 and n_outputs > 0:
        for x in op.inputs:
            if x.name not in blob_to_optype_output_dict:
                blob_to_optype_output_dict[x.name] = (x,op.type,op.outputs[0])
            

for keys in blob_to_optype_output_dict:
    name = keys
    
    ctw = 0
    if len(blob_to_optype_output_dict[name][0].get_shape().as_list()) != 4:
        for i in range(10):
            if name in blob_to_optype_output_dict and ctw < 1:
                if len(blob_to_optype_output_dict[name][0].get_shape().as_list()) == 4:
                    ctw += 1
                current_node = blob_to_optype_output_dict[name][0]
                op_type = blob_to_optype_output_dict[name][1]
                print current_node.get_shape().as_list(),
                print ' ---> (', op_type, ') ---> ',
                name = blob_to_optype_output_dict[name][2].name
            else:
                break    
    
    print '\n'
    
    
                   
        


