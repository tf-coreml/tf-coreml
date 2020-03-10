[![Build Status](https://travis-ci.com/apple/tfcoreml.svg?branch=master)](#)
[![PyPI Release](https://img.shields.io/pypi/v/tfcoreml.svg)](#)
[![Python Versions](https://img.shields.io/pypi/pyversions/tfcoreml.svg)](#)

Convert from Tensorflow to CoreML
=================================

Python package to convert from Tensorflow to CoreML format. To get the latest
version of `tfcoreml`, please run:

```shell
pip install --upgrade tfcoreml
```

For the latest changes please see the [release
notes](https://github.com/tf-coreml/tf-coreml/releases).

Usage
------

Please see the Tensorflow conversion section in the [Neural network
guide](https://github.com/apple/coremltools/blob/master/examples/NeuralNetworkGuide.md)
on how to use the converter. 

There are several [notebook
examples](https://github.com/apple/coremltools/tree/master/examples/neural_network_inference)
as well for reference.  

There are scripts in the `utils/` directory for visualizing and writing out a
text summary of a given frozen TensorFlow graph.  This could be useful in
determining the input/output names and shapes.  Another useful tool for
visualizing frozen TensorFlow graphs is
[Netron](https://github.com/lutzroeder/Netron).

## License
[Apache License 2.0](LICENSE)
