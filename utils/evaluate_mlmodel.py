import numpy as np


image_input = np.random.rand(1, 256, 256, 3)
index_input = np.random.rand(26)

output_name = ['scaled_transformer/contract/conv1/InstanceNorm/moments/StopGradient:0',
               'transformer/contract/conv1/InstanceNorm/moments/SquaredDifference:0_difference',
               'transformer/contract/conv1/InstanceNorm/moments/SquaredDifference:0',
               'transformer/contract/conv1/InstanceNorm/moments/variance:0'  
               ]

'''
Evaluate CoreML
'''
import coremltools

image_input_coreml = np.expand_dims(image_input, axis = 0) #(1,1,256,256,3)
image_input_coreml = np.transpose(image_input_coreml, (0,1,4,2,3)) #(1,1,3,256,256)
index_input_coreml = np.reshape(index_input, (1,1,1,1,26)) #(seq, batch, C, H, W) == (1,1,1,1,26)

mlmodel = coremltools.models.MLModel('style.mlmodel')

if 0:
    spec = mlmodel.get_spec()
    for out in output_name:
        #add the output name
        new_output = spec.description.output.add()
        new_output.name = out
        new_output_params = new_output.type.multiArrayType
        new_output_params.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value('DOUBLE')
        coremltools.utils.save_spec(spec, "style.mlmodel")

    mlmodel = coremltools.models.MLModel('style.mlmodel')

coreml_input = {'input:0': image_input_coreml, 'style_num:0': index_input_coreml}
coreml_pred = mlmodel.predict(coreml_input)
for out in output_name:
    coreml_out = coreml_pred[out]
    print 'coreml_out: ', out , ' : ', coreml_out.shape, coreml_out.flatten()[:10]

