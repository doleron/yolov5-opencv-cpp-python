import cv2
import matplotlib
import numpy
import numpy as np
import caffe
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import imageio

net = cv2.dnn.readNetFromCaffe('caffee/model/deploy.prototxt', 'caffee/model/0_iter_106500.caffemodel')


model_def = 'caffee/model/deploy.prototxt'
model_weights = 'caffee/model/0_iter_106500.caffemodel'


# net = caffe.Net(model_def,      # defines the structure of the model
#                 model_weights,  # contains the trained weights
#                 caffe.TEST)     # use test mode (e.g., don't perform dropout)

mu = np.load('caffee/model/mean.npy')
mu = mu.mean(1).mean(1)
print('mean-subtracted values:', zip('BGR', mu))
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
# transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

f = 'caffee/images/0db0c337db0a3a4df796e9eafa6c02ed.jpg'
# print(f)
# image = caffe.io.load_image(f)
#
# transformed_image = transformer.preprocess('data', image)
# sizechanged_img = transformer.deprocess('data',transformed_image)

# copy the image data into the memory allocated for the net
# net.blobs['data'].data[...] = transformed_image

### perform classification
net.setInput(cv2.dnn.blobFromImage(transformed_image, 1 / 255.0, image.shape, swapRB=True, crop=False))
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the image
cls = output_prob.argmax()

