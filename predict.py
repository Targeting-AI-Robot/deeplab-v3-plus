import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from model import Deeplabv3
import argparse
import scipy.misc


parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, help='Image to process segmentation')
args = parser.parse_args()

# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

trained_image_width=512 
mean_subtraction_value=127.5
inputfile = args.img
image = np.array(Image.open(inputfile))

# resize to max dimension of images from training dataset
w, h, _ = image.shape
ratio = float(trained_image_width) / np.max([w, h])
resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

# apply normalization for trained dataset images
resized_image = (resized_image / mean_subtraction_value) - 1.

# pad array to square image to match training images
pad_x = int(trained_image_width - resized_image.shape[0])
pad_y = int(trained_image_width - resized_image.shape[1])
resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

# make prediction
deeplab_model = Deeplabv3()
res = deeplab_model.predict(np.expand_dims(resized_image, 0))
labels = np.argmax(res.squeeze(), -1)

# remain only people label number 15
labels[labels!=15] = 0

# remove padding and resize back to original image
if pad_x > 0:
    labels = labels[:-pad_x]
if pad_y > 0:
    labels = labels[:, :-pad_y]
labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))

# make output img
labels[labels!=15] = 0
labels[labels==15] = 255

#plt.imshow(labels)
outputfile = inputfile.split(".")[0]+"_out.png"
im = Image.fromarray(labels)
im.save(outputfile)
#plt.waitforbuttonpress()
