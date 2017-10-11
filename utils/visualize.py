import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import shutil
import sys

def visualize(model_filename):

    if os.path.exists("/tmp/pb"):
        shutil.rmtree('/tmp/pb')  

    with tf.Session() as sess:
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    LOGDIR='/tmp/pb'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)


if __name__ == "__main__":
    if len(sys.argv) != 2: 
        print "Usage: visualize tf_network.pb"
    # load file
    visualize(sys.argv[1])
    print("launch the command: ")
    print('tensorboard --logdir=/tmp/pb')    
    
# after executing this script, execute the following command in terminal: 
# tensorboard --logdir=/tmp/pb

# to kill tensorboard process, for opening a new graph, use the following commands in the terminal
# ps aux | grep tensorboard
# kill PID

