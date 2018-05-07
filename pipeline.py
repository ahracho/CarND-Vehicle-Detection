import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import time
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label
import os
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from util_function import *


colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (32, 32)
hist_bins = 32
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"


def process_image(img):
    ystart = int(img.shape[0] * 0.55)
    ystop = int(img.shape[0] * 0.85)
    scales = [1.1, 1.5]
    threshold = 1
    hot_window = []
    for scale in scales:
        hot_window.extend(find_cars(img, colorspace, ystart, ystop, scale, svc,
                                    X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    marked_image = heatmap_detection(img, hot_window, threshold)
    return marked_image


if os.path.isfile("svc_pickle.p"):
    with open("svc_pickle.p", mode='rb') as f:
        print("Use existing pickle file....")
        svc_pickle = pickle.load(f)
        svc = svc_pickle["svc"]
        X_scaler = svc_pickle["scaler"]
else:
    print("Start training classifier....")
    svc, X_scaler = training_classifier(colorspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)

    svc_pickle = {}
    svc_pickle["svc"] = svc
    svc_pickle["scaler"] = X_scaler
    pickle.dump(svc_pickle, open("svc_pickle.p", "wb"))


print("Start Converting Video....")

output_video = 'final_video.mp4'
clip1 = VideoFileClip("./project_video.mp4")
output_clip = clip1.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)

