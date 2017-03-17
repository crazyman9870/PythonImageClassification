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

'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread(args["image"])

# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
print("[INFO] loading and preprocessing image...")
image = image_utils.load_img(args["image"], target_size=(224, 224))
image = image_utils.img_to_array(image)


# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network -- we'll also preprocess the image by
# subtracting the mean RGB pixel intensity from the ImageNet dataset
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet")

# classify the image
print("[INFO] classifying image...")
preds = model.predict(image)
x = decode_predictions(preds)[0]
'''

with open('labels.csv', 'w') as csvfile:
	counter = 1
	for file in os.listdir('images'):
		#print('\nhere',file)
		file = 'images\\' + file
		# load the original image via OpenCV so we can draw on it and display
		# it to our screen later
		orig = cv2.imread(file)

		# load the input image using the Keras helper utility while ensuring
		# that the image is resized to 224x224 pxiels, the required input
		# dimensions for the network -- then convert the PIL image to a
		# NumPy array

		#print("[INFO] loading and preprocessing image...")
		image = image_utils.load_img(file, target_size=(224, 224))
		image = image_utils.img_to_array(image)


		# our image is now represented by a NumPy array of shape (3, 224, 224),
		# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
		# pass it through the network -- we'll also preprocess the image by
		# subtracting the mean RGB pixel intensity from the ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

		# load the VGG16 network

		#print("[INFO] loading network...")
		model = VGG16(weights="imagenet")

		# classify the image

		#print("[INFO] classifying image...")
		preds = model.predict(image)
		x = decode_predictions(preds, 40)[0]

		#print('PREDS\n\n',x,'\n\n')
		inID = []
		label = []
		percent = []

		for x_ in x:
			inID.append(x_[0])
			label.append(x_[1])
			percent.append(x_[2])
		#(inID, label, percent) = x[0]

		csvfile.write(file)
		csvfile.write('\n')
		csvfile.write(','.join(str(l) for l in label))
		csvfile.write('\n')
		csvfile.write(','.join(str(p) for p in percent))
		csvfile.write('\n')
		'''
		# display the predictions to our screen
		print("1 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[0], label[0], percent[0]))
		print("2 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[1], label[1], percent[1]))
		print("3 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[2], label[2], percent[2]))
		print("4 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[3], label[3], percent[3]))
		print("5 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[4], label[4], percent[4]))
		
		print("6 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[5], label[5], percent[5]))
		print("7 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[6], label[6], percent[6]))
		print("8 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[7], label[7], percent[7]))
		print("9 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[8], label[8], percent[8]))
		print("10 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[9], label[9], percent[9]))
		print("11 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[10], label[10], percent[10]))
		print("12 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[11], label[11], percent[11]))
		print("13 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[12], label[12], percent[12]))
		print("14 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[13], label[13], percent[13]))
		print("15 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[14], label[14], percent[14]))
		print("16 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[15], label[15], percent[15]))
		print("17 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[16], label[16], percent[16]))
		print("18 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[17], label[17], percent[17]))
		print("19 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[18], label[18], percent[18]))
		print("20 ImageNet ID: {}, Label: {}, Percent: {}".format(inID[19], label[19], percent[19]))
		'''
		if counter % 10 == 0:
			print('\n\nProcessed 10 images')
			print(counter)
			print('\n\n')
			'''
			cv2.putText(orig, "Label: {}".format(label[0]), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
			cv2.imshow("Classification", orig)
			cv2.waitKey(0)
			'''
		counter += 1
	print('Finished')