# tfcoreml
TensorFlow (TF) to CoreML Converter

Dependencies
-------------

- tensorflow >= 1.1.0
- coremltools >= 0.6
- numpy >= 1.6.2
- protobuf >= 3.1.0
- six==1.10.0

## Installation

### Install From PyPI
To install Pypi package:

```
pip install -U tfcoreml
```

### Install From Source

To build and install from source:
```
python setup.py bdist_wheel
```

This will generate a pip installable wheel inside the `dist` directory. 

To install as a package with `pip` : at the root directory, run:
```
pip install -e .
```

## Usage

See iPython notebooks in the directory `examples/` for examples of
how to use the converter.

The following arguments are required by the CoreML converter:
- path to the frozen .pb graph file to be converted
- path where the .mlmodel should be written
- a list of output tensor names present in the TF graph
- a dictionary of input names and their shapes (as list of integers). 
  This is only required if input tensors' shapes are not fully defined in the frozen .pb file 
	(e.g. they contain `None` or `?`)

Note that the frozen .pb file can be obtained from the checkpoint and graph def files
by using the `tensorflow.python.tools.freeze_graph` utility. 
For details of freezing TF graphs, please refer to the TensorFlow documentation and the notebooks in directory `examples/` in this repo.
There are scripts in the `utils/` directory for visualizing and writing out a text summary of a given frozen TF graph. This could be useful in determining the input/output names and shapes.  

**For example:**

When input shapes are fully determined in the frozen .pb file:
```
import tfcoreml as tf_converter
tf_converter.convert(tf_model_path = 'my_model.pb',
                     mlmodel_path = 'my_model.mlmodel',
                     output_feature_names = ['softmax:0'])					
```

When input shapes are not fully specified in the frozen .pb file:
```
import tfcoreml as tf_converter
tf_converter.convert(tf_model_path = 'my_model.pb',
                     mlmodel_path = 'my_model.mlmodel',
                     output_feature_names = ['softmax:0'],
                     input_name_shape_dict = {'input:0' : [1, 227, 227, 3]})
```


## Supported Ops

List of TensorFlow ops that are supported currently (see `tfcoreml/_ops_to_layers.py`): 

* Add
* ArgMax
* AvgPool
* BatchNormWithGlobalNormalization
* BatchToSpaceND*
* BiasAdd
* ConcatV2, Concat
* Const
* Conv2D
* Conv2DBackpropInput
* DepthwiseConv2dNative
* Elu
* Exp
* ExtractImagePatches
* Fill*
* FloorMod*
* FusedBatchNorm
* Gather*
* Greater*
* GreaterEqual*
* Identity
* Log
* LogicalAnd*
* LRN
* MatMul
* Max*
* Maximum
* MaxPool
* Mean*
* Min*
* Minimum
* MirrorPad
* Mul
* Neg
* OneHot
* Pad
* Placeholder
* Prod*
* RandomUniform*
* RealDiv
* Reciprocal
* Relu
* Relu6
* Reshape*
* ResizeNearestNeighbor
* Rsqrt
* Shape
* Sigmoid
* Slice
* Softmax
* SpaceToBatchND*
* Square
* SquaredDifference
* StridedSlice
* Sub
* Sum*
* Tanh
* Transpose*

Note that certain parameterizations of these ops may not be fully supported. For ops marked with an asterisk, only very specific usage patterns are supported. In addition, there are several other ops (not listed above) that are skipped by the converter as they generally have no effect during inference. Kindly refer to the files `tfcoreml/_ops_to_layers.py` and `tfcoreml/_layers.py` for full details. 


Scripts for converting several of the following pretrained models can be found at `tests/test_pretrained_models.py`. 
Other models with similar structures and supported ops can be converted. 
Below is a list of publicly available TensorFlow frozen models that can be converted with this converter:

- [Inception v1 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz)
- [Inception v2 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_2016_08_28_frozen.pb.tar.gz)
- [Inception v3 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz)
- [Inception v4 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz)
- [Inception v3 (non-Slim)*](https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip) 
- [Inception/ResNet v2 (Slim)](https://storage.googleapis.com/download.tensorflow.org/models/inception_resnet_v2_2016_08_30_frozen.pb.tar.gz)
- MobileNet variations (Slim):
  - Image size: 128 ([1](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.25_128_frozen.tgz), 
                      [2](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz), 
                      [3](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_128_frozen.tgz), 
                      [4](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_128_frozen.tgz))
  - Image size: 160 ([1](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.25_160_frozen.tgz), 
                      [2](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_160_frozen.tgz), 
                      [3](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_160_frozen.tgz), 
                      [4](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_160_frozen.tgz))
  - Image size: 192 ([1](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.25_192_frozen.tgz), 
                      [2](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_192_frozen.tgz), 
                      [3](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_192_frozen.tgz), 
                      [4](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_192_frozen.tgz))
  - Image size: 224 ([1](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.25_224_frozen.tgz), 
                      [2](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_224_frozen.tgz), 
                      [3](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_224_frozen.tgz), 
                      [4](
                      https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz))                                                                  
- [Image stylization network+](https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip)
- [Mobilenet SSD*](https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip) 

*Converting these models requires extra steps to extract subgraphs from the TF frozen graphs. See `examples/` for details. <br>
+There are known issues running image stylization network on GPU. (See Issue #26)


### Limitations

`tfcoreml` converter has the following constraints: 

- TF graph must be cycle free (cycles are generally created due to control flow ops like `if`, `while`, `map`, etc.)
- Must have `NHWC` ordering (Batch size, Height, Width, Channels) for image feature map tensors
- Must have tensors with rank less than or equal to 4 (`len(tensor.shape) <= 4`)
- The converter produces CoreML model with float values. A quantized TF graph (such as the style transfer network linked above) gets converted to a float CoreML model

## Running Unit Tests

In order to run unit tests, you need pytest.

```shell
pip install pytest
```

To add a new unit test, add it to the `tests/` folder. Make sure you
name the file with a 'test' as the prefix.
To run all unit tests, navigate to the `tests/` folder and run

```shell
pytest
```

## Directories
- "tfcoreml": the tfcoreml package
- "examples": examples to use this converter
- "tests": unit tests
- "utils": general scripts for graph inspection

## License
[Apache License 2.0](LICENSE)
