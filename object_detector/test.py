import argparse as ap
import os

import cv2
import matplotlib.pyplot as plt
import scipy.misc
from skimage.feature import hog
from sklearn.externals import joblib
#SS -> 2 lines
from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from color_histogram.core.hist_3d import Hist3D

from config import *
from utils import sliding_window, pyramid, rgb2gray


class Detector:
	def __init__(self, downscale=1.5, window_size=(178, 218), window_step_size=32, threshold=0.4):
		self.clf = joblib.load(MODEL_PATH)
		self.downscale = downscale
		self.window_size = window_size
		self.window_step_size = window_step_size
		self.threshold = threshold

	def detect(self, image):
		clone = image.copy();

		image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
		image1_hist = cv2.calcHist([image1], [0], None, [256], [0, 256]);
		#print(image1_hist[1])

		# list to store the detections
		detections = [];
		# current scale of the image
		downscale_power = 0;
		no_of_windows = 0;
		no_of_windows_cc = 0;
		w = self.window_size[1] * self.window_size[0];
		# downscale the image and iterate
		for im_scaled in pyramid(image, downscale=self.downscale, min_size=self.window_size):
			# if the width or height of the scaled image is less than
			# the width or height of the window, then end the iterations
			if im_scaled.shape[0] < self.window_size[1] or im_scaled.shape[1] < self.window_size[0]:
				break
			for (x, y, im_window) in sliding_window(im_scaled, self.window_step_size, self.window_size):
				if im_window.shape[0] != self.window_size[1] or im_window.shape[1] != self.window_size[0]:
					continue
				no_of_windows = no_of_windows + 1;
				#im_window1 = rgb2gray(im_window)
				im_window1 = cv2.cvtColor(im_window, cv2.COLOR_BGR2GRAY);
				im_window1_hist = cv2.calcHist([im_window1], [0], None, [256], [0, 256]);
				ss_sum = 0;
				rows,cols = im_window1.shape;
				for i in range(rows):
					for j in range(cols):
						pixel = im_window1[i][j];
						ss_sum = ss_sum + min(abs((image1_hist[pixel] - im_window1_hist[pixel])),abs(im_window1_hist[pixel]));
				#for pixel in getPixels(im_window1):
					
				#print(ss_sum);
				ss_sum = ss_sum / w;
				'''
				print("ss_sum");
				print(ss_sum);
				SS = (1 - (ss_sum));
				print("SS of window")
				print(SS)
				'''
				#img = img_as_float(im_window)
				#segments_slic = slic(img, n_segments=10, compactness=10, sigma=1)
				#print(segments_slic)
				im_window_cc = image[y:y + 50 + self.window_size[1], x:x + 50 +self.window_size[0]]
				im_window_lab = cv2.cvtColor(im_window,cv2.COLOR_BGR2LAB);
				im_window_cc_lab =  cv2.cvtColor(im_window_cc,cv2.COLOR_BGR2LAB);
				channels = cv2.split(im_window_lab)       # Set the image channels
				colors = ("l", "a", "b")        # Initialize tuple
				for (i, col) in zip(channels, colors):       # Loop over the image channels
					im_window_hist = cv2.calcHist([i], [0], None, [256], [0, 256])   # Create a histogram for current channel
				
				channels = cv2.split(im_window_cc_lab)
					
				for (i, col) in zip(channels, colors):       # Loop over the image channels
					im_window_cc_hist = cv2.calcHist([i], [0], None, [256], [0, 256])   # Create a histogram for current channel
 
 
				coeff=cv2.compareHist(im_window_hist,im_window_cc_hist,cv2.cv2.HISTCMP_CHISQR);
				#if (ss_sum > 200):
				if (coeff > 1000 or ss_sum > 200):#smaller the COEFF value, silimar are the images. We want difference in color hence, bigger numbers
					
					feature_vector = hog(im_window1)
					X = np.array([feature_vector])
					prediction = self.clf.predict(X)
					if prediction == 1:
						x1 = int(x * (self.downscale ** downscale_power))
						y1 = int(y * (self.downscale ** downscale_power))
						detections.append((x1, y1,x1 + int(self.window_size[0] * (self.downscale ** downscale_power)),y1 + int(self.window_size[1] * 								(self.downscale ** downscale_power))))
						no_of_windows_cc = no_of_windows_cc + 1
			downscale_power += 1
		print("Number of windows without CC")
		print(no_of_windows)
		
		print("Number of windows with cc + ss")
		print(no_of_windows_cc)
		clone_detected = clone.copy()
		for (x1, y1, x2, y2) in detections:
			cv2.rectangle(clone_detected, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

		return clone_detected


if __name__ == '__main__':
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', '--images_dir_path', help='Path to the test images dir',
                        required=True)
    parser.add_argument('-v', '--visualize', help='Visualize the sliding window',
                        action='store_true')
    args = vars(parser.parse_args())

    visualize_det = args['visualize']
    image_dir_path = args['images_dir_path']

    detector = Detector(downscale=PYRAMID_DOWNSCALE, window_size=WINDOW_SIZE,
                        window_step_size=WINDOW_STEP_SIZE, threshold=THRESHOLD)

    for image_name in os.listdir(image_dir_path):
        if image_name == '.DS_Store':
            continue

        # Read the image
        image = scipy.misc.imread(os.path.join(image_dir_path, image_name))

        # detect faces and return detected image
        image_detected = detector.detect(image)

        plt.imshow(image_detected)
        plt.xticks([]), plt.yticks([])
        plt.show()

