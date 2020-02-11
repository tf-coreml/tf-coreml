# tfcoreml - TensorFlow to Core ML Converter

Tool to convert models defined in Tensorflow to the Core ML format. 

To get the latest version of `tfcoreml` from PyPI:

```shell
pip install --upgrade tfcoreml
pip install --upgrade coremltools # tfcoreml depends on the coremltools package
```

For the latest changes please see the [release notes](https://github.com/tf-coreml/tf-coreml/releases).


Usage
------

Please see the Tensorflow conversion section in the [Neural network guide](https://github.com/apple/coremltools/blob/master/examples/NeuralNetworkGuide.md) 
on how to use the converter. 

There are several [notebook examples](https://github.com/apple/coremltools/tree/master/examples/neural_network_inference) 
as well for reference.  

There are scripts in the `utils/` directory for visualizing and writing out a text summary of a given frozen TensorFlow graph.
This could be useful in determining the input/output names and shapes.
Another useful tool for visualizing frozen TensorFlow graphs is [Netron](https://github.com/lutzroeder/Netron).

Dependencies
------------

- tensorflow >= 1.5.0
- coremltools >= 3.1

## Installation

### Install from Source

To get the latest version of the converter, clone this repository and install from source:

```shell
git clone https://github.com/tf-coreml/tf-coreml.git
cd tf-coreml
```

To install as a package with `pip`, either run (at the root directory):

```shell
pip install -e .
```

or run:

```shell
python setup.py bdist_wheel
```

This will generate a `pip` installable wheel inside the `dist/` directory.

### Install From PyPI

```shell
pip install --upgrade tfcoreml
```


## Running Unit Tests

In order to run unit tests, you need `pytest`.

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

- `tfcoreml`: the tfcoreml package
- `examples`: examples to use this converter
- `tests`: unit tests
- `utils`: general scripts for graph inspection

## License
[Apache License 2.0](LICENSE)
