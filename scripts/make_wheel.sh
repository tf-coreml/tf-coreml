#!/bin/bash
set -e

BUILD_DIR=build
PYTHON_ENV=$BUILD_DIR/tf-coreml-env
PYTHON=$(which python)

unknown_option() {
  echo "Unknown option $1. Exiting."
  exit 1
}

print_help() {
  echo "Test the wheel by running all unit tests"
  echo
  echo "Usage: ./test_wheel.sh --wheel-path=WHEEL_PATH"
  echo
  echo "  --wheel-path=*          Specify which wheel to test."
  echo "  --python=*              Python to use for configuration."
  echo
  exit 1
} # end of print help

# Command flag options
while [ $# -gt 0 ]
  do case $1 in
    --python=*)          PYTHON=${1##--python=} ;;
    --wheel-path=*)      WHEEL_PATH=${1##--wheel-path=} ;;
    --help)              print_help ;;
    *) unknown_option $1 ;;
  esac
  shift
done

# Describe the python version we are using
echo "Configuring using python from $PYTHON"

#  Setup the right python & pip executable
$PYTHON -m pip install virtualenv
$PYTHON -m virtualenv $PYTHON_ENV
source $PYTHON_ENV/bin/activate
PYTHON=$(which python)
PIP=$PYTHON_ENV/bin/pip

# Make the wheel
$PIP install -r requirements.pip --upgrade
$PYTHON setup.py bdist_wheel 
