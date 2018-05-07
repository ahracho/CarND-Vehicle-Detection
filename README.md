# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repository is for the project of Udacity Nanodegree - Self-driving Car Engineer : Vehicle Detection Proejct. It is forked from (https://github.com/udacity/CarND-Vehicle-Detection).

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

It needs us to apply everything we learned from the lecture including Decision Tree, SVM to train classifier. Several methods to remove false positives. Details on the pipeline is described in `writeup.md`.


This project has 3 types of outputs:

1. final_video.mp4 : final video with vehicle detected
2. pipeline.py : script used to produce final_video.mp4
~~~sh
python pipeline.py
~~~
3. writeup.md : writeup file that specify details on how I completed this project.
---
4. util_function.py : dependent files for executing pipeline.py
5. svc_pickle.p : pickle data to save SVC model and Scaler (but failed to upload on github due to over-size(160MB))

* For now, SVC classifier works pretty slow, and spends lots of time to convert video, so I would try training classifier with CNN model and optimize the performance.  
