#from mlkit_proto import Model_pb2
from google.protobuf import text_format
import time
import coremltools

path = '../models/ResNet/cifar10/resnet.mlmodel'
#READ
model = coremltools.utils.load_spec(path)
#model = Model_pb2.Model()
f = open(path, "rb")
g = f.read()
t0 = time.time()
model.ParseFromString(g)
t1 = time.time()
f.close()
print 'time to execute ParseFromString', (t1-t0), 'secs'


#Write back as txt
f = open("/tmp/resnet.txt", "w")
t0 = time.time()
f.write(text_format.MessageToString(model))
f.close()
t1 = time.time()
print 'time to write text file', (t1-t0), 'secs'
