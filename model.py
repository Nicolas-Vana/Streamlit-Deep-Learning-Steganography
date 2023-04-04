from pathlib import Path
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
from skimage.metrics import structural_similarity as ssim
import streamlit as st
import matplotlib.pyplot as plt
import math


import tensorflow as tf
import tensorflow_wavelets.Layers.DWT as DWT
from tensorflow import keras


# root = Path(os.getcwd())
# hide = root / 'hide'
# reveal = root / 'reveal'

# Loss function used in the models. Required to load the model
def get_custom_loss(y_true, y_pred):
  gain = 0
  gain += tf.image.psnr(y_true, y_pred, 1.)
  gain += tf.image.ssim(y_true, y_pred, 1.)

  return -1*gain

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_net(path):
    model = tf.keras.models.load_model(path / 'Model', custom_objects={'get_custom_loss': get_custom_loss})
    return model

# Tensorflow model that does the Discrete Wavelet Transform of an image.
def get_DWT_TF(tensor, scale):

    # Compute the DWT of each of the channels of an RGB image.
    scope_name = 'DWT'
    dwt_0 = DWT.DWT(name=scope_name + "0",concat=0)(tf.reshape(tensor[:,:,:,0], (1, scale, scale, 1)))
    dwt_1 = DWT.DWT(name=scope_name + "1",concat=0)(tf.reshape(tensor[:,:,:,1], (1, scale, scale, 1)))
    dwt_2 = DWT.DWT(name=scope_name + "2",concat=0)(tf.reshape(tensor[:,:,:,2], (1, scale, scale, 1)))

    # We then stack all of the DWTs of each of the channels into one array
    tensor_stacked = tf.stack([dwt_0, dwt_1, dwt_2], axis = -1)

    # Define the model
    DWT_net = keras.Model(
    inputs=[tensor],
    outputs=[tensor_stacked],
    name = 'DWT_net',
    )

    return DWT_net

# Tensorflow model that does the Inverse Discrete Wavelet Transform of an image.
def get_IDWT_TF(tensor, scale):

  # Concatenate each of the approximation and detail images of the DWT, for each color channel
  channels = []
  for channel in range(3):
    tensor_channel = tensor[:,:,:,:,channel]
    ll = tf.reshape(tensor_channel[:,:,:,0], (1, int(scale/2), int(scale/2)))
    lh = tf.reshape(tensor_channel[:,:,:,1], (1, int(scale/2), int(scale/2)))
    hl = tf.reshape(tensor_channel[:,:,:,2], (1, int(scale/2), int(scale/2)))
    hh = tf.reshape(tensor_channel[:,:,:,3], (1, int(scale/2), int(scale/2)))

    channels.append(tf.reshape(tf.concat([tf.concat([ll, lh], axis=1), tf.concat([hl, hh], axis=1)], axis=2), (1, scale, scale)))

  tmp = tf.convert_to_tensor(channels)
  concat_tensor = tf.stack([tmp[0,:,:,:], tmp[1,:,:,:], tmp[2,:,:,:]], axis=-1)

  # IDWT layers on each color channel
  scope_name = 'hide_IDWT'
  idwt_tensor_0 = DWT.IDWT(name = scope_name + "0", splited = 0)(tf.reshape(concat_tensor[:,:,:,0], (1, scale, scale, 1), 'reshape_channel_0'))
  idwt_tensor_1 = DWT.IDWT(name = scope_name + "1", splited = 0)(tf.reshape(concat_tensor[:,:,:,1], (1, scale, scale, 1), 'reshape_channel_1'))
  idwt_tensor_2 = DWT.IDWT(name = scope_name + "2", splited = 0)(tf.reshape(concat_tensor[:,:,:,2], (1, scale, scale, 1), 'reshape_channel_2'))

  # And then we stack the channels and reshape to yield a image with dimensions (1, scale, scale, 3).
  tensor_stacked = tf.stack([idwt_tensor_0, idwt_tensor_1, idwt_tensor_2], axis=-1, name='hide_stack')
  tensor_idwt = tf.reshape(tensor_stacked, (1, scale, scale, 3), name = 'final_reshape_idwt')

  # Define the model
  IDWT_net = keras.Model(
    inputs=[tensor],
    outputs=[tensor_idwt],
    name = 'IDWT_NET',
    )

  return IDWT_net

# Runs the Hide model.
def hide(model, cover, secret, batch_size, scale):
    # creates filler data to accomodate for the batch size of the pre-trained model, and appends to the cover and secret image. This is innefficient but is not too costly.
    filler_data = np.random.normal(size = (batch_size-1,scale,scale,3))
    cover = np.reshape(cover, (1, scale, scale, 3))
    secret = np.reshape(secret, (1, scale, scale, 3))

    covers = np.append(cover, filler_data, axis = 0)
    secrets = np.append(secret, filler_data, axis = 0)

    # Run model and return resulting stego image
    stegos = model.predict([covers, secrets], batch_size = batch_size, verbose = 0)
    return stegos[0]

# Runs the Reveal model.
def reveal(model, stego, batch_size, scale):
    # creates filler data to accomodate for the batch size of the pre-trained model, and appends to the stego image. This is innefficient but is not too costly.
    filler_data = np.random.normal(size = (batch_size-1,int(scale/2),int(scale/2),4,3))
    stegos = np.append(stego, filler_data, axis = 0)

    # Run model and return resulting recovered secret image
    recovered_secrets = model.predict([stegos], batch_size = batch_size)
    return recovered_secrets[0]

# Used to compute stats for nerds
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))
