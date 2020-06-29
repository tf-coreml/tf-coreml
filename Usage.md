Usage
-----

To convert models trained/saved via TensorFlow 1, first export them into the frozen graph def format, which is a protobuf file
format with `.pb` as the extension. Frozen `.pb` files can be obtained by using TensorFlow's
`tensorflow.python.tools.freeze_graph` utility.

[This](https://github.com/apple/coremltools/blob/3.4/examples/neural_network_inference/tensorflow_converter/Tensorflow_1/linear_mnist_example.ipynb) Jupyter notebook shows how to freeze a graph to produce a `.pb` file.

There are several other Jupyter notebook examples for conversion 
[here](https://github.com/apple/coremltools/tree/3.4/examples/neural_network_inference/tensorflow_converter/Tensorflow_1).

```python
import tfcoreml

tfcoreml.convert(tf_model_path='my_model.pb',
                 mlmodel_path='my_model.mlmodel',
                 output_feature_names=['softmax:0'],  # name of the output tensor (appended by ":0")
                 input_name_shape_dict={'input:0': [1, 227, 227, 3]},  # map from input tensor name (placeholder op in the graph) to shape
                 minimum_ios_deployment_target='12') # one of ['12', '11.2']
```