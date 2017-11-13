# tfcoreml
TensorFlow (TF) to CoreML Converter

Dependencies
-------------

- tensorflow >= 1.1.0
- coremltools >= 0.6
- numpy >= 1.6.2
- protobuf >= 3.1.0
- six==1.10.0

## Installation:

To build wheel:
```
python setup.py bdist_wheel
```

This will generate a pip installable wheel inside the `dist` directory. 

To install as a package with `pip` : at the root directory, run:
```
pip install -e .
```

## Usage:

See iPython notebooks in the directory `examples/` for examples of
how to use the converter.

More specifically, following arguments are required by the CoreML converter:
- path to the frozen .pb graph file to be converted
- path where the .mlmodel should be written
- a list of output tensor names present in the TF graph
- a dictionary of input names and their shapes (as list of integers), 
  if input tensors' shape is not fully defined in the frozen .pb file 
	(e.g. contains `None` or `?`)

Note that the frozen .pb file can be obtained from the checkpoint and graph def files
by using the `tensorflow.python.tools.freeze_graph` utility. 
For details of freezing TF graphs, please refer to TensorFlow documentation and the notebooks in directory `examples/` in this repo. 

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


### Supported Ops and Models

For a list of supported TF operations and their parameters please refer to `tfcoreml/_ops_to_layers.py`. 

Scripts for converting the following pretrained models can be found at `tests/test_pretrained_models.py`. 
Other models with similar structures and supported ops can be converted. 
Below is a list of publicly TensorFlow models that can be converted with this converter:

- [Inception v3 (non-Slim)*](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip) 

- [Inception v1 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz)

- [Inception v2 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_2016_08_28_frozen.pb.tar.gz)

- [Inception v3 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz)

- [Inception v4 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz)

- [Inception/ResNet v2 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_resnet_v2_2016_08_30_frozen.pb.tar.gz)

- MobileNet variations (Slim) 
  - [[1]](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.25_128_frozen.tgz)
  - [[2]](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz)
  - [[3]](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_128_frozen.tgz)

- [Image stylization network+]('https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip')

*Converting these models require extra steps to extract subgraphs from the TF frozen graphs. See `examples/` for details. 
+There're still open issues on running image stylization network on GPU. (See Issue #26)


### Limitations:

`tfcoreml` converter has the following constraints: 

- TF graph should not contain cycles (which are generally due to control flow ops like `if`, `while`, `map`, etc.)
- Must have `NHWC` ordering (Batch size, Height, Width, Channels) for image feature map tensors
- Must not contain tensors with rank greater than 4 (`len(tensor.shape) <= 4`)
- The converter produces CoreML model with float values. A quantized TF graph (such as the style transfer network linked above) gets converted to a float CoreML model. 

## Directories:
- "tfcoreml": the tfcoreml package
- "examples": examples to use this converter
- "tests": unit tests
- "utils": general utils for evalaution and graph inspection

