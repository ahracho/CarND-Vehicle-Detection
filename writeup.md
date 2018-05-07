**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car_not_car.png
[image2]: ./writeup_images/hog_car.png
[image3]: ./writeup_images/hog_notcar.png
[image4]: ./writeup_images/boxed_image.png
[image5]: ./writeup_images/heatmap.png
[image6]: ./writeup_images/heatmap_threshold.png
[image7]: ./writeup_images/label.png
[image8]: ./writeup_images/label_boxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

My project includes the following files:
* pipeline.py           : execution script for detecting vehicles on the road
* final_video.mp4   : final output video (based on project_video.mp4)
* writeup.md          : summarizing the results


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `get_hog_features()` function in util_function.py.   

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:  

![Car/Not Car Image][image1]  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  

Here is an example using the `YCrCb` color space and HOG parameters of `orient=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:  

This is hog-featured image of a vehicle shown above.  
![alt text][image2]  

This is hog-featured image of a tree shown above.  
![alt text][image3]  

#### 2. Explain how you settled on your final choice of HOG parameters.

For me, finding the best combination of parameters which leads both high accuracy and short training time was the first thing to get through. I tried various combinations of parameters and my final choice is `orient=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.   

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To extract features, I used all of the information I get from the images : Spatial features, color histogram features and hog features for all channels. Extracting feature is described in `extract_features()` function in util_function.py. For each of the image in data-set, at first I converted images into `YCrCb`, and then extracted spatial / color histogram / hog features. I though the more features I use, the more accurate model can be trained. Totally, I used 10224 features for each image.  

I tried several methods to train the classifier: DecisionTree, SVC with linear kernel and SVC with RBF kernel. All methods lead to the classifier which of the accuracy is higher than 97%. Initially, I thought 97% of accuracy is high enough to get the good result, but with processing video frames, it causes lots of false-positive predictions. My final choice is SVC with RBF kernel. It has accuracy of 99.75%. It is much slower than Linear SVC but the number of false positive predictions were also dropped significantly.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is implemented in `find_cars()` function in util_function.py. I could have slided windows and extract features for each step, but especially for hog features, it is quite redundant process. I extracted hog featues just once for entire image rather than calling `get_hog_features()` each by each. (Line 198-203).

And then, I slided 64x64 window(Line 187) and extracted features for each window. Window is slided 2 cells per step(overlapped 75% because it has 8 cells per block). Scale of image has been changed. I tried several combinations and chose scale of 1.1 and 1.5 for each image (pipeline.py Line 30).  

Since the classifier I chose was working relatively slow, I wanted to reduce the area window moved around. Window is moved within ystart(`int(img.shape[0] * 0.55) = 396`) and ystop(`int(img.shape[0] * 0.85) = 612`).  

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales(1.1 and 1.5) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.

Following detection logic described above, boxes predicted as vehicles are marked.  
![Box Marked][image4]

To remove false positive, and draw a rectangular around the vehicle, I used heatmap and label method. Details are described in 2nd question in Video Implementation section.

Counting how many times pixels are predicted as vehicle, and apply threshold(`add_heat()` and `apply_threshold()` in util_function.py).   
![HeatMap][image5]  
![HeatMap with threshold][image6]  

After applying threshold (I used 1 for threshold) onto heatmap, we have to group them up with label() method. With label function we can label heatmap as unique features. Labeled boxes can be regarded as the final decision of vehicle detection.  
![Label Gray][image7]  
![Label Boxes][image8]  

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final_video.mp4)  
There are few false positives I cannot finally removed, but most of time in video stream, classifier successfully dectects vehicles.  

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

One of biggest barriers I faced while doing this project was reducing false positives. Most of the classifier I tried succeeded to detect vehicles but they also had lots of false positives.   

At first, I tried DecisionTree and its accuracy was not high enough(stays around 97-98%).  

Next, I tried LinearSVC. It has higher accuracy(around 98-98.8%) and is good for detecting vehicles. But it has too much false positives espcially with shadow in the image. I added more training data for non-vehicle data set, but it still had many false positives. I tried filtering final windows based on its size. For example, boxes smaller than 100 pixels are not counted as final detection. It helped to filter small wandering boxes, but I still got false positives (20-22 second in project_video.mp4). I tried various C value but result was same. Of course, I tried heatmap method to filter false positives, but it didn't work that well. Since classifier itself makes wrong decision too much, the effect of filtering methods was not good enough.  
I was stick to use LinearSVC because it was the fastest classifier to process video. I tried everything to tune LinearSVC and reduce false positives but it all failed. It was time to move on.

Finally, I changed classifier to SVC with RBF kernel. It was much slower compared to LinearSVC, but accuracy was much higher(higher than 99% most of time). With C parameter 10, I got 99.8% of accuracy, and surprisingly the number of false positives dropped dramatically. I filtered remain false positives with window size (Line 274 in util_function.py, windows smaller than 50pxs are counted) and decision_function (Line 236 in util_function.py, result of the decision_function should be larger than 0.3).

I also add heatmap logic to make sure filtering false positives(`heatmap_detection()` in util_function.py). After sliding windows with different scales, draw a heatmap and adjust threshold to figure out which positions are most likely to predict as a vehicle. With label(), I can get final result of vehicle detection.

Sample images are attached above.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I am not satisfied with my classifier yet. Although it predicts vehicle well enough but process takes too much time. If it is real world situation, this system would be too slow to prevent a car accident. So after submission, I would try this project with CNN layer again. In the student community, many people succeed vehicle detection with CNN classifer. I hope with NN classifier, process time would be reduced dramatically while keeping the accuracy.
