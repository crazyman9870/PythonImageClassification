#from joblib import Parallel, delayed
#import multiprocessing

from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16

import numpy as np
import argparse
import os
import csv

def getLabels(image_location):
    loc = open(image_location, 'r')

    image = image_utils.load_img(loc.name, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    model = VGG16(weights='imagenet')

    preds = model.predict(image)
    predictions = decode_predictions(preds, 1000)[0]
    predictions.sort(key=lambda tup: tup[1])

    percentages = [x[2] for x in predictions]

    loc.close()

    return percentages

def writeCSV(file_name, data):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    f.close()


if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--images', required=True, help='path to the folder with input images')
	ap.add_argument('-r', '--results', required=True, help='path to the csv with results')
	ap.add_argument('-l', '--labels', required=True, help='csv file name that will store image file names')
	args = vars(ap.parse_args())



	images = []
	image_locations = []
	for root, dirs, filenames in os.walk(args['images']):
	    for f in filenames:
	        file_location = os.path.join(root, f)
	        images.append((f, file_location))
	        image_locations.append(file_location)

	#num_cores = multiprocessing.cpu_count()
	predictions = []
	for filepath in image_locations:
		predictions.append(getLabels(filepath))

	writeCSV(args['results'], predictions)
	writeCSV(args['labels'], images)
