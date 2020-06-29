[![Build Status](https://travis-ci.com/apple/tfcoreml.svg?branch=master)](#)
[![PyPI Release](https://img.shields.io/pypi/v/tfcoreml.svg)](#)
[![Python Versions](https://img.shields.io/pypi/pyversions/tfcoreml.svg)](#)

Convert from Tensorflow to CoreML
=================================

[coremltools](https://github.com/apple/coremltools) (Recommended approach)
--------------------


For converting TensorFlow models to CoreML format, the recommended approach is to use TensorFlow converter available through **new** unified conversion API, introduced in`coremltools 4.0` python package.
Please read the coremltools documentation on [Tensorflow conversion](https://coremltools.readme.io/docs/tensorflow-conversion) for example usage.

To install coremltools package, please follow [these instructions](https://coremltools.readme.io/docs/installation) in the coremltools documentation.


tfcoreml 
---------

`tfcoreml` package is **no longer maintained**. 

Conversion API `tfcoreml.convert` should **only be used** if **all** of the following conditions are met:
 1. Primary deployment target is `iOS 12` or earlier. 
 2. Source model is a TensorFlow 1 `graph_def` object serialized as frozen protobuf format (".pb") 
 
 
 To install `tfcoreml`, please run:

```shell
pip install --upgrade tfcoreml
```

Please read [this usage section](./Usage.md) which illustrates how to convert models using `tfcoreml`.

For access to new features, bug fixes, community support and requests, please use [coremltools](https://github.com/apple/coremltools) github repository.


## License
[Apache License 2.0](LICENSE)
