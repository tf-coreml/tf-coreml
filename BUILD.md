### Install from Source

To get the latest version of the converter, clone this repository and install
from source. 

First you need to install the dependencies 
```
pip install -r requirements.pip
```

Then you can install the package from the source directory.
```shell
pip install -e .
```

## Running Unit Tests

In order to run unit tests, you need `pytest`. To add a new unit test, add it
to the `tests/` folder. Make sure you name the file with a 'test' as the
prefix.  To run all unit tests, navigate to the `tests/` folder and run

```shell
pip install -e .
pytest -ra tests/
```

### Building wheels

If you want to build a wheel for distribution or testing, you may run 

```shell
python setup.py bdist_wheel
```
This will generate a `pip` installable wheel inside the `dist/` directory.
There is a script that automates all of the steps necessary for building a
wheel (including installing the right depedencies), `scripts/make_wheel.sh`,
that can be used instead.

## License
[Apache License 2.0](LICENSE)
