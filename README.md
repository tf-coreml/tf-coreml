# tfcoreml
TensorFlow (TF) to CoreML Converter

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

More specifically, provide these as CoreML converter inputs:
- path to the frozen pb file to be converted
- path where the .mlmodel should be written
- a list of output tensor names
- a dictionary of input names and their shapes (as list of integers), 
  if input tensors' shape is not fully determined in the frozen .pb file 
	(e.g. contains `None` or `?`)

Note that the frozen .pb file can be obtained from the checkpoint files
by using `tensorflow.python.tools.freeze_graph` utility. 
For details of freezing TF graphs, please refer to TensorFlow documentation.

### Limitations:
The current version of `tfcoreml` can only convert a TensorFlow graph that:
graph: 
- do not contain control-flow ops, like `if`, `while`, `map`, etc.;
- do not contain cycles
- uses `NHWC` (Batch size, Height, Width, Channels) for image feature maps
- contains tensors whose rank is no greater than 4 (`len(tensor.shape) <= 4`)
- contains only tensors of float types

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
- "tfcoreml": the tfcoreml package
- "examples": examples to use this converter
- "tests": unittests
- "utils": general utils for evaluation and graph inspection
