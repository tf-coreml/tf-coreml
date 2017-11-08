import unittest
import urllib
import os, sys
import tarfile, zipfile
from os.path import dirname
import numpy as np
import PIL.Image
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import coremltools
import tfcoreml as tf_converter

TMP_MODEL_DIR = '/tmp/tfcoreml'
TEST_IMAGE = './test_images/beach.jpg' 

def _download_file(url):
  """Download the file.
  url - The URL address of the frozen file
  fname - Filename of the frozen TF graph in the url. 
  """
  dir_path = TMP_MODEL_DIR
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)
  
  k = url.rfind('/')
  fname = url[k+1:]
  fpath = os.path.join(dir_path, fname)

  ftype = None
  if url.endswith(".tar.gz") or url.endswith(".tgz"):
    ftype = 'tgz'
  elif url.endswith('.zip'):
    ftype = 'zip'

  if not os.path.exists(fpath):
    urllib.urlretrieve(url, fpath)
  if ftype == 'tgz':
    tar = tarfile.open(fpath)
    tar.extractall(dir_path)
    tar.close()
  elif ftype == 'zip':
    zip_ref = zipfile.ZipFile(fpath, 'r')
    zip_ref.extractall(dir_path)
    zip_ref.close()

def _compute_max_relative_error(x,y):
  rerror = 0
  index = 0
  for i in range(len(x)):
    den = max(1.0, np.abs(x[i]), np.abs(y[i]))
    if np.abs(x[i]/den - y[i]/den) > rerror:
      rerror = np.abs(x[i]/den - y[i]/den)
      index = i
  return rerror, index  
    
def _compute_SNR(x,y):
  noise = x - y
  noise_var = np.sum(noise ** 2)/len(noise) + 1e-7
  signal_energy = np.sum(y ** 2)/len(y)
  max_signal_energy = np.amax(y ** 2)
  SNR = 10 * np.log10(signal_energy/noise_var)
  PSNR = 10 * np.log10(max_signal_energy/noise_var)   
  return SNR, PSNR     

def _load_image(path, resize_to=None):
  img = PIL.Image.open(path)
  if resize_to is not None:
    img = img.resize(resize_to, PIL.Image.ANTIALIAS)
  img_np = np.array(img).astype(np.float32)
  return img_np, img

def _generate_data(input_shape, mode = 'random'):
  """
  Generate some random data according to a shape.
  """
  if input_shape is None or len(input_shape) == 0:
    return 0.5
  if mode == 'zeros':
    X = np.zeros(input_shape)
  elif mode == 'ones':
    X = np.ones(input_shape)
  elif mode == 'linear':
    X = np.array(range(np.product(input_shape))).reshape(input_shape)*1.0
  elif mode == 'random':
    X = np.random.rand(*input_shape)
  elif mode == 'random_zero_mean':
    X = np.random.rand(*input_shape)-0.5
  elif mode == 'image':
    # Load a real image and do default tf imageNet preprocessing
    img_np, _ = _load_image(TEST_IMAGE ,resize_to=(img_size, img_size))
    img_tf = np.expand_dims(img_np, axis = 0)
    X = img_tf * 2.0/255 - 1
  elif mode == 'onehot_0':
    X = np.zeros(input_shape)
    X[0] = 1
  return X

def _tf_transpose(x, is_sequence=False):
  if not hasattr(x, "shape"):
    return x
  if len(x.shape) == 4:
    # [Batch, Height, Width, Channels] --> [Batch, Channels, Height, Width]
    x = np.transpose(x, [0,3,1,2])
    return np.expand_dims(x, axis=0)
  elif len(x.shape) == 3:
    # We only deal with non-recurrent networks for now
    # [Batch, (Sequence) Length, Channels] --> [1,B, Channels, 1, Seq]
    # [0,1,2] [0,2,1]
    return np.transpose(x, [0,2,1])[None,:,:,None,:]
  elif len(x.shape) == 2:
    if is_sequence:  # (N,S) --> (S,N,1,)
      return x.reshape(x.shape[::-1] + (1,))
    else:  # (N,C) --> (N,C,1,1)
      return x.reshape((1, ) + x.shape) # Dense
  elif len(x.shape) == 1:
    if is_sequence: # (S) --> (S,N,1,1,1)
      return x.reshape((x.shape[0], 1, 1))
    else: 
      return x
  else:
    return x

class CorrectnessTest(unittest.TestCase):
  
  @classmethod
  def setUpClass(self):
    """ Set up the unit test by loading common utilities.
    """
    self.err_thresh = 0.15
    self.snr_thresh = 15
    self.psnr_thresh = 30
    self.red_bias = -1
    self.blue_bias = -1
    self.green_bias = -1
    self.image_scale = 2.0/255
    
  def _compare_tf_coreml_outputs(self, tf_out, coreml_out):
    self.assertEquals(len(tf_out), len(coreml_out))
    error, ind = _compute_max_relative_error(coreml_out, tf_out)    
    SNR, PSNR = _compute_SNR(coreml_out, tf_out)
    self.assertGreater(SNR, self.snr_thresh)
    self.assertGreater(PSNR, self.psnr_thresh)
    self.assertLess(error, self.err_thresh)


  def _test_tf_model(self, tf_model_path, coreml_model, input_tensors, 
      output_tensor_names, data_modes = 'random', delta = 1e-2, 
      use_cpu_only = False):
    """ Common entry to testing routine (Tensors in, tensors out).
    tf_model_path - frozen TF model path
    coreml_model - MLModel object
    input_tensors -  list of (name,shape) for each input (placeholder)
    output_tensor_names - output_tensor_names, a list of strings
    """
    # Load TensorFlow model
    tf.reset_default_graph()
    graph_def = graph_pb2.GraphDef()
    with open(tf_model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    g = tf.import_graph_def(graph_def)
    
    if type(data_modes) is str:
      data_modes = [data_modes] * len(input_tensors)

    with tf.Session(graph = g) as sess:
      # Prepare inputs
      feed_dict = {}
      for idx, in_tensor in enumerate(input_tensors):
        ts_name, ts_shape = in_tensor
        feed_dict[ts_name] = _generate_data(ts_shape, data_mode[idx])
      # Run TF session
      result = sess.run(output_tensor_names, feed_dict=feed_dict)

    # Evaluate coreml model
    coreml_inputs = {}
    for idx, in_tensor in enumerate(input_tensors):
      in_tensor_name, in_shape = in_tensor
      colon_pos = in_tensor_name.rfind(':')
      coreml_in_name = in_tensor_name[:colon_pos] + '__0'
      coreml_inputs[coreml_in_name] = _tf_transpose(
          feed_dict[in_tensor_name]).copy()
        
    coreml_output = coreml_model.predict(coreml_inputs, useCPUOnly=use_cpu_only)
    
    for idx, out_name in enumerate(output_tensor_names):
      tp = _tf_transpose(result[idx]).flatten()
      colon_pos = out_name.rfind(':')
      out_tensor_name = out_name[:colon_pos] + '__0'
      cp = coreml_output[out_tensor_name].flatten()
      self.assertEquals(len(tp), len(cp))
      for i in xrange(len(tp)):
        max_den = max(1.0, tp[i], cp[i])
        self.assertAlmostEquals(tp[i]/max_den, cp[i]/max_den, delta=delta)


  def _test_coreml_model_image_input(self, tf_model_path, coreml_model, 
      input_tensor_name, output_tensor_name, img_size):
    """Test single image input conversions.
    tf_model_path - the TF model
    coreml_model - converted CoreML model
    input_tensor_name - the input image tensor name
    output_tensor_name - the output tensor name
    img_size - size of the image
    """

    img_np, img = _load_image(TEST_IMAGE ,resize_to=(img_size, img_size))
    img_tf = np.expand_dims(img_np, axis = 0)
    img_tf[:,:,:,0] = self.image_scale * img_tf[:,:,:,0] + self.red_bias 
    img_tf[:,:,:,1] = self.image_scale * img_tf[:,:,:,1] + self.green_bias 
    img_tf[:,:,:,2] = self.image_scale * img_tf[:,:,:,2] + self.blue_bias 
    
    #evaluate the TF model
    tf.reset_default_graph()
    graph_def = graph_pb2.GraphDef()
    with open(tf_model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    g = tf.import_graph_def(graph_def)
    with tf.Session(graph=g) as sess:  
      image_input_tensor = sess.graph.get_tensor_by_name('import/' + input_tensor_name)
      output = sess.graph.get_tensor_by_name('import/' + output_tensor_name)
      tf_out = sess.run(output,feed_dict={image_input_tensor: img_tf})
    if len(tf_out.shape) == 4:
      tf_out = np.transpose(tf_out, (0,3,1,2))
    tf_out_flatten = tf_out.flatten()
    
    #evaluate CoreML
    coreml_input_name = input_tensor_name.replace(':', '__')
    coreml_output_name = output_tensor_name.replace(':', '__')
    coreml_input = {coreml_input_name: img}
    
    #Test by forcing CPU evaluation
    coreml_out = coreml_model.predict(coreml_input, useCPUOnly = True)[coreml_output_name]
    coreml_out_flatten = coreml_out.flatten()
    self._compare_tf_coreml_outputs(tf_out_flatten, coreml_out_flatten)
    
    #Test the default CoreML evaluation
    coreml_out = coreml_model.predict(coreml_input)[coreml_output_name]
    coreml_out_flatten = coreml_out.flatten()
    self._compare_tf_coreml_outputs(tf_out_flatten, coreml_out_flatten)

class TestModels(CorrectnessTest):         
  
  def test_inception_v3_slim(self):
    #Download model
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz'
    tf_model_dir = _download_file(url = url)
    tf_model_path = os.path.join(TMP_MODEL_DIR, 'inception_v3_2016_08_28_frozen.pb')
    
    #Convert to coreml
    mlmodel_path = os.path.join(TMP_MODEL_DIR, 'inception_v3_2016_08_28.mlmodel')
    mlmodel = tf_converter.convert(
        tf_model_path = tf_model_path,
        mlmodel_path = mlmodel_path,
        output_feature_names = ['InceptionV3/Predictions/Softmax:0'],
        input_name_shape_dict = {'input:0':[1,299,299,3]},
        image_input_names = ['input:0'],
        red_bias = -1, 
        green_bias = -1, 
        blue_bias = -1, 
        image_scale = 2.0/255.0)

    #Test predictions on an image
    self._test_coreml_model_image_input(
        tf_model_path = tf_model_path, 
        coreml_model = mlmodel,
        input_tensor_name = 'input:0',
        output_tensor_name = 'InceptionV3/Predictions/Softmax:0',
        img_size = 299)

  def test_mobilenet_v1_25_128(self):
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.25_128_frozen.tgz'
    tf_model_dir = _download_file(url = url)
    tf_model_path = os.path.join(TMP_MODEL_DIR, 'mobilenet_v1_0.25_128/frozen_graph.pb')

    mlmodel_path = os.path.join(TMP_MODEL_DIR, 'mobilenet_v1_0.25_128.mlmodel')
    mlmodel = tf_converter.convert(
        tf_model_path = tf_model_path,
        mlmodel_path = mlmodel_path,
        output_feature_names = ['MobilenetV1/Predictions/Softmax:0'],
        input_name_shape_dict = {'input:0':[1,128,128,3]},
        image_input_names = ['input:0'],
        red_bias = -1, 
        green_bias = -1, 
        blue_bias = -1, 
        image_scale = 2.0/255.0)

    #Test predictions on an image
    self._test_coreml_model_image_input(
        tf_model_path = tf_model_path, 
        coreml_model = mlmodel,
        input_tensor_name = 'input:0',
        output_tensor_name = 'MobilenetV1/Predictions/Softmax:0',
        img_size = 128)
        
        
  def test_mobilenet_v1_100_224(self):
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz'
    tf_model_dir = _download_file(url = url)
    tf_model_path = os.path.join(TMP_MODEL_DIR, 'mobilenet_v1_1.0_224/frozen_graph.pb')

    mlmodel_path = os.path.join(TMP_MODEL_DIR, 'mobilenet_v1_1.0_224.mlmodel')
    mlmodel = tf_converter.convert(
        tf_model_path = tf_model_path,
        mlmodel_path = mlmodel_path,
        output_feature_names = ['MobilenetV1/Predictions/Softmax:0'],
        input_name_shape_dict = {'input:0':[1,224,224,3]},
        image_input_names = ['input:0'],
        red_bias = -1, 
        green_bias = -1, 
        blue_bias = -1, 
        image_scale = 2.0/255.0)

    #Test predictions on an image
    self._test_coreml_model_image_input(
        tf_model_path = tf_model_path, 
        coreml_model = mlmodel,
        input_tensor_name = 'input:0',
        output_tensor_name = 'MobilenetV1/Predictions/Softmax:0',
        img_size = 224)
        
  def test_mobilenet_v1_75_192(self):
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.75_192_frozen.tgz'
    tf_model_dir = _download_file(url = url)
    tf_model_path = os.path.join(TMP_MODEL_DIR, 'mobilenet_v1_0.75_192/frozen_graph.pb')

    mlmodel_path = os.path.join(TMP_MODEL_DIR, 'mobilenet_v1_0.75_192.mlmodel')
    mlmodel = tf_converter.convert(
        tf_model_path = tf_model_path,
        mlmodel_path = mlmodel_path,
        output_feature_names = ['MobilenetV1/Predictions/Softmax:0'],
        input_name_shape_dict = {'input:0':[1,192,192,3]},
        image_input_names = ['input:0'],
        red_bias = -1, 
        green_bias = -1, 
        blue_bias = -1, 
        image_scale = 2.0/255.0)

    #Test predictions on an image
    self._test_coreml_model_image_input(
        tf_model_path = tf_model_path, 
        coreml_model = mlmodel,
        input_tensor_name = 'input:0',
        output_tensor_name = 'MobilenetV1/Predictions/Softmax:0',
        img_size = 192) 
        
  def test_mobilenet_v1_50_160(self):
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_160_frozen.tgz'
    tf_model_dir = _download_file(url = url)
    tf_model_path = os.path.join(TMP_MODEL_DIR, 'mobilenet_v1_0.50_160/frozen_graph.pb')

    mlmodel_path = os.path.join(TMP_MODEL_DIR, 'mobilenet_v1_0.50_160.mlmodel')
    mlmodel = tf_converter.convert(
        tf_model_path = tf_model_path,
        mlmodel_path = mlmodel_path,
        output_feature_names = ['MobilenetV1/Predictions/Softmax:0'],
        input_name_shape_dict = {'input:0':[1,160,160,3]},
        image_input_names = ['input:0'],
        red_bias = -1, 
        green_bias = -1, 
        blue_bias = -1, 
        image_scale = 2.0/255.0)

    #Test predictions on an image
    self._test_coreml_model_image_input(
        tf_model_path = tf_model_path, 
        coreml_model = mlmodel,
        input_tensor_name = 'input:0',
        output_tensor_name = 'MobilenetV1/Predictions/Softmax:0',
        img_size = 160)                 
  
  #@unittest.skip("Failing: related to https://github.com/tf-coreml/tf-coreml/issues/26")
  def test_style_transfer(self):
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip'
    tf_model_dir = _download_file(url = url)
    tf_model_path = os.path.join(TMP_MODEL_DIR, 'stylize_quantized.pb')
    mlmodel_path = os.path.join(TMP_MODEL_DIR, 'stylize_quantized.mlmodel')
    # ? style transfer image size and style number?
    mlmodel = tf_converter.convert(
        tf_model_path = tf_model_path,
        mlmodel_path = mlmodel_path,
        output_feature_names = ['Squeeze:0'],
        input_name_shape_dict = {'input:0':[1,128,128,3], 'style_num:0':[26]})

    # Test predictions on an image
    input_tensors = [('import/input:0',[1,256,256,3]), 
                     ('import/style_num:0',[26])]

    self._test_tf_model(
        tf_model_path = tf_model_path,
        coreml_model = mlmodel,
        input_tensors = input_tensors,
        output_tensor_names = ['import/Squeeze:0'],
        data_modes = ['image', 'onehot_0'], 
        delta = 1e-2,
        use_cpu_only = False)
      