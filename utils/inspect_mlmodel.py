import coremltools
import sys, os
from infer_shapes_nn_mlmodel import infer_shapes
import operator

def inspect(model_path):
    spec = coremltools.utils.load_spec(model_path)
    
    sys.stdout = open(os.devnull, "w")
    shape_dict = infer_shapes(model_path)
    sys.stdout = sys.__stdout__
    
    types_dict = {}
    
    nn = spec.neuralNetwork
    for i, layer in enumerate(nn.layers):
        print('---------------------------------------------------------------------------------------------------------------------------------------------')
        print("{}: layer name = {}, layer type = ( {} ), \n inputs = \n {}, \n input shapes = {}, \n outputs = \n {}, \n output shapes = {} ".format(i, layer.name, \
                layer.WhichOneof('layer'), ", ".join([x for x in layer.input]), ", ".join([str(shape_dict[x]) for x in layer.input]), 
                ", ".join([x for x in layer.output]),
                ", ".join([str(shape_dict[x]) for x in layer.output])))
                
                
        layer_type = layer.WhichOneof('layer')        
        if layer_type in types_dict:
            types_dict[layer_type] += 1
        else:
            types_dict[layer_type] = 1
    
    print('---------------------------------------------------------------------------------------------------------------------------------------------')
    sorted_types_count = sorted(types_dict.items(), key=operator.itemgetter(1))           
    print('Layer Type counts:')
    for i in sorted_types_count:
        print("{} : {}".format(i[0], i[1]))        
                
    
if __name__ == "__main__":
    if len(sys.argv) != 2: 
        print "Usage: inspect_mlmodel.py network.mlmodel"
    # load file
    inspect(sys.argv[1])                 