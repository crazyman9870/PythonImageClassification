# USAGE
# python test_imagenet.py --image images/dog_beagle.png

# import the necessary packages
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import numpy as np
import argparse
import cv2
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to the folder with input images")
args = vars(ap.parse_args())

image_labels = {}

for root, dirs, filenames in os.walk(args["images"]):
	for f in filenames:

		log = open(os.path.join(root, f), 'r')
		print(str(log.name))


		# load the original image via OpenCV so we can draw on it and display
		# it to our screen later
		orig = cv2.imread(log.name)

		# load the input image using the Keras helper utility while ensuring
		# that the image is resized to 224x224 pxiels, the required input
		# dimensions for the network -- then convert the PIL image to a
		# NumPy array
		# print("[INFO] loading and preprocessing image...")
		image = image_utils.load_img(log.name, target_size=(224, 224))
		image = image_utils.img_to_array(image)

		# our image is now represented by a NumPy array of shape (3, 224, 224),
		# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
		# pass it through the network -- we'll also preprocess the image by
		# subtracting the mean RGB pixel intensity from the ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

		# load the VGG16 network
		# print("[INFO] loading network...")
		model = VGG16(weights="imagenet")

		# classify the image
		# print("[INFO] classifying image...")
		preds = model.predict(image)
		predictions = decode_predictions(preds)[0]
		image_labels[log.name] = predictions

		# # display the predictions to our screen
		# (inID, label, percent) = predictions[0]
		# print("ImageNet ID: {}, Label: {}, Percent: {}".format(inID, label, percent))
		# cv2.putText(orig, "Label: {}".format(label), (10, 30),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
		# cv2.imshow("Classification", orig)
		# cv2.waitKey(0)
		# for prediction in predictions[1:]:
		# 	(inID, label, percent) = prediction
		# 	print("ImageNet ID: {}, Label: {}, Percent: {}".format(inID, label, percent))

print(str(image_labels))