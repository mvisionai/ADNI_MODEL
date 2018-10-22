
import  numpy as np
import argparse
from tensorflow.keras.preprocessing.image import  load_img
data = np.array([[1, 2, 3],[6,56,9],[4,10,45]])

di=np.take(data, np.random.permutation(len(data)), axis=0, out=data)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

print(args)
