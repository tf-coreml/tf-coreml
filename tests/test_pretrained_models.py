import unittest
import urllib
import os, sys
import tarfile
import ipdb
from os.path import dirname
import numpy as np
import PIL.Image
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import coremltools
import tfcoreml as tf_converter

TMP_MODEL_DIR = '/tmp/tfcoreml'
TEST_IMAGE = './test_images/beach.jpg' 

def _download_file(url, fname):
  dir_path = TMP_MODEL_DIR
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)
  fpath = os.path.join(dir_path, fname)     
  
  url_is_tar_gz = False
  if url.endswith("tar.gz"):
    url_is_tar_gz = True  
  fpath_full = fpath + '.tar.gz' if url_is_tar_gz else fpath
  
  if not os.path.exists(fpath):
    urllib.urlretrieve(url, fpath_full)
    if url_is_tar_gz:  
      tar = tarfile.open(fpath_full)
      tar.extractall(dir_path)
      tar.close()
      os.remove(fpath_full)
      
  return fpath 
  
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
      
    
    
  def _test_coreml_model_image_input(self, tf_model_path, coreml_model, 
      input_tensor_name, output_tensor_name, img_size):
    
    def _load_image(path, resize_to=None):
      img = PIL.Image.open(path)
      if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
      img_np = np.array(img).astype(np.float32)
      return img_np, img
      
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
    sess = tf.Session(graph = g)
    image_input_tensor = sess.graph.get_tensor_by_name(
        'import/' + input_tensor_name)
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
    coreml_out = coreml_model.predict(coreml_input, 
        useCPUOnly = True)[coreml_output_name]
    coreml_out_flatten = coreml_out.flatten()
    self._compare_tf_coreml_outputs(tf_out_flatten, coreml_out_flatten)
    
    #Test the default CoreML evaluation
    coreml_out = coreml_model.predict(coreml_input)[coreml_output_name]
    coreml_out_flatten = coreml_out.flatten()
    self._compare_tf_coreml_outputs(tf_out_flatten, coreml_out_flatten)
    
        
class TestModels(CorrectnessTest):         
  
  def test_inception_v3_slim(self):
    
    #Download model
    fname = 'inception_v3_2016_08_28_frozen.pb'
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz'
    tf_model_path = _download_file(url = url, fname = fname)
    
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
    print('Conversion Done. Now testing......')
    self._test_coreml_model_image_input(
        tf_model_path = tf_model_path, 
        coreml_model = mlmodel,
        input_tensor_name = 'input:0',
        output_tensor_name = 'InceptionV3/Predictions/Softmax:0',
        img_size = 299)                                   
                                        
                                          
                                 
  
    
    
    