## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The dataset was broken down to two sets, car and notcar, in the beginning of the project, I used the sample images that were provided by Udacity.

I also visualized some of the images, from both the classes, that can be seen in the IPython Notebook

Then I used the helper functions that were taught us during the lecture videos, as you can see under the hedding of "Necessary Functions for feature extraction" in the IPython Notebook.

I also used the color histograms technique and explored a few color spaces to extract the necessary features from the images.

I used the skimage.hog() function, and I tuned the some of the parameters namely
> orientations - Specifies the number of gradient directions 
> pixels_per_cell - The size of the cell over which the gradient was calculated
> cells_per_block - area over which the histogram count is normalized

I also, visualized the HOG for some of the sample images, which can be seen in the IPython Notebook

#### 2. Explain how you settled on your final choice of HOG parameters.

There were a lot of parameters that could be tunes, I did some manual tuning on the images and finally settled for 

* color_space = 'YCrCb' - YCrCb resulted in far better performance than RGB, HSV and HLS
* orient = 9 # HOG orientations - I tried 6,9 and 12. Model performance didn't vary much
* pix_per_cell = 16 - I tried 8 and 16 and finally chose 16 since it signficantly decreased computation time
* cell_per_block = 1 - I tried 1 and 2. The performance difference b/w them wasn't much but 1 cell per block had significantly less no. of features and speeded up training and pipeline
* hog_channel = 'ALL' - ALL resulted in far better performance than any other individual channel

These parameters, were quick in their performance as well as accuracy, as ultimately these parameters have to be used in a video, therefor we have to keep these things in mind

Also, after a lot of tuning, although there were combinations that were able to yield a better images on the test set, but on the video, they were not actually performing very well. So these were the parameters that I decided to finally choose.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I followed the steps below for training the classifier

1) Format features using np.vstack and StandardScaler().
2) Split data into shuffled training and test sets
3) Train linear SVM using sklearn.svm.LinearSVC().


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

As there was not a need to actually run the algorithm on the whole image, I reduced the area where the sliding window algorithm will actually run, 
In the sliding window technique, for each window we extract features for that window, scale extracted features to be fed to the classifier, predict whether the window contains a car using our trained Linear SVM classifier and save the window if the classifier predicts there is a car in that window.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Refer to the test images, that are displayed in the IPython Notebook, under the heading "Visualization of the sliding window technique on test images"


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./vehicle_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach actually, used the classic machine learning approach, which as told in the videos was nice to get a good intuition as it requires a lot of manual tuning, but what I felt was that because of the excess manual tuning, perhaps, the parameters the I selected are way too specific to this problem.
Also when I tried to adjust so as to balance the speed of the algorithm a lot of false positives started popping up

I think with the introduction of Deep Nets like YOLOnet, we can leverage the power of Deep Learning here too, to get better and quicker results. Which actually have been mention in many places
