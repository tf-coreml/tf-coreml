import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

np.set_printoptions(precision=3)


graph_def = graph_pb2.GraphDef()
with open("style_opt.pb", "rb") as f:
    graph_def.ParseFromString(f.read())
g = tf.import_graph_def(graph_def)


sess = tf.Session(graph = g)

image_input = sess.graph.get_tensor_by_name("import/input:0")
index_input = sess.graph.get_tensor_by_name("import/style_num:0")

output = sess.graph.get_tensor_by_name("import/transformer/expand/conv3/conv/Sigmoid:0")

y_tf = sess.run(output,feed_dict={image_input: np.random.rand(1, 256, 256, 3), index_input: np.random.rand(26)})
print y_tf.shape
print y_tf.flatten()[:5]