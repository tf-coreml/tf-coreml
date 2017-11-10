# tfcoreml
TensorFlow (TF) to CoreML Converter

Dependencies: 
tensorflow >= 1.1.0, 
coremltools >= 0.6
numpy >= 1.6.2
protobuf >= 3.1.0
six==1.10.0

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

See iPython notebook examples in `examples/` for demonstrations about
how to use this converter.

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

- does not contain control-flow ops, like `if`, `while`, `map`, etc.;
- does not contain cycles
- uses `NHWC` ordering (Batch size, Height, Width, Channels) for image feature maps
- contains tensors whose rank is no greater than 4 (`len(tensor.shape) <= 4`)
- contains only tensors of float types
- contains unsupported TF operations - Please refer to `_ops_to_layers.py` 
    for a list of supported ops

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

### Supported Models

The following standard models can be converted. Other models with similar structures can also be converted. 

- [GoogLeNet v3 (non-Slim)*](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip) 

- [GoogLeNet v1 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz)

- [GoogLeNet v2 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_2016_08_28_frozen.pb.tar.gz)

- [GoogLeNet v3 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz)

- [GoogLeNet v4 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz)

- [GoogLeNet/ResNet v2 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_resnet_v2_2016_08_30_frozen.pb.tar.gz)

- MobileNet variations (Slim) [[1]](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.25_128_frozen.tgz)
    [[2]](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz)
		[[3]](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_128_frozen.tgz)

*Converting these models require extra steps to extract subgraphs from TF
models. See `examples/` for details. 


## Directories:
- "tfcoreml": the tfcoreml package
- "examples": examples to use this converter
- "tests": unit tests
- "utils": general utils for evalaution and graph inspection

