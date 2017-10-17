# tfcoreml
TensorFlow to CoreML Converter

dependencies: tensorflow >= 1.1.0, coremltools >= 0.6

## Installation:

To build wheel: 
```
python setup.py bdist_wheel
```

To install as a package with `pip`: at the root directory, run:
```
pip install -e .
```

## Usage:
See `examples/linear_mnist_example.ipynb` that demonstrate how to use this converter. 
The converter only supports Tensorflow models that use the 'NHWC' format. 

More specifically, provide:
- path to the frozen pb file to be converted 
- path where the .mlmodel should be written
- a list of output tensor names 
- you may need to provide a dictionary of input names and their shapes (as list of integers), if input tensors' shape is not fully determined in the frozen .pb file (e.g. contains `None` or `?`)

Note that the frozen .pb file can be obtained from the checkpoint files
by using the freeze_graph.py script that comes with Tensorflow. Please refer to Tensorflow documentation.

inspect_pb.py file in the utils folder can be used to infer output tensor names 
(the "import" prefix in the name of the tensors can be dropped). 

e.g.: 

When input shapes are fully determined in frozen .pb file:
```
import tfcoreml as tf_converter
tf_converter.convert(tf_model_path = 'my_model.pb', 
                     mlmodel_path = 'my_model.mlmodel', 
                     output_feature_names = ['softmax:0'])					
```

When input shapes are not fully determined in frozen .pb file:
```
import tfcoreml as tf_converter
tf_converter.convert(tf_model_path = 'my_model.pb',
                     mlmodel_path = 'my_model.mlmodel',
                     output_feature_names = ['softmax:0'], 
                     input_name_shape_dict = {'input:0' : [1, 227, 227, 3]})
```

## Directories:
- "examples": examples to use this converter
- "tests": unittests
- "utils": general utils for evalaution and graph inspection
