# tf-coreml
TensorFlow to CoreML Converter

dependencies: tensorflow >= 1.1.0, coremltools >= 0.6

USAGE:

Provide:
- path to the frozen pb file to be converted 
- path where the .mlmodel should be written
- a list of output tensor names 
- you may need to provide a dictionary of input names and their shapes (as list of integers), if the input shape information cannot be gathered from the frozen .pb file

Note that the frozen .pb file can be obtained from the checkpoint files
by using the freeze_graph.py script that comes with Tensorflow. Please refer to Tensorflow documentation.

inspect_pb.py file in the utils folder can be used to infer output tensor names 
(the "import" prefix in the name of the tensors can be dropped). 

The converter only supports Tensorflow models that use the 'NHWC' format. 

e.g.: 

```
	import tf_converter
	tf_converter.convert(tf_model_path = 'my_model.pb', mlmodel_path = 'my_model.mlmodel', 
							output_feature_names = ['softmax:0'])					
```
```
	import tf_converter
	tf_converter.convert(tf_model_path = 'my_model.pb', mlmodel_path = 'my_model.mlmodel', 
							output_feature_names = ['softmax:0'], input_name_shape_dict = {'input:0' : [1, 227, 227, 3]})					
```


To build wheel: 
```
	python setup.py bdist_wheel
```

"tests" directory: unittests

"models" directory: place to store .pb, .mlmodels and the comparison scripts

"utils" directory: general utils for evalaution and graph inspection