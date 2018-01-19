## Self Driving Car Nanodegreee
### Vehicle Detection and Tracking

---

[//]: # (Image References)
[image1]: ./project_images/combo3in1_a.jpg
[image2]: ./project_images/combo3in1_b.jpg
[image3]: ./project_images/combo3in1_b.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./output_videos/test_video.mp4
[video2]: ./output_videos/output_video.mp4

---
### Introduction

This is the writeup for the final project of term 1 of the Self Driving Car Nanodegree.

The code for this project can be seen with the associated ipython notebook.

```sh
P5_vehicle_detection.ipynb
```

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Test Video

Here's a [link to my test video output][video1]

As we can se above, both cars are successfully detected.


#### 2. Project Video

Here's a [link to my project video output][video2]


#### 3. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

The code to do the above is represented below.

```sh
from scipy.ndimage.measurements import label

ystart1 = 380
ystop1 = 686
scale1 = 1.2

ystart2 = 350
ystop2 = 650
scale2 = 1.5

ystart3 = 350
ystop3 = 590
scale3 = 1.7

ystart4 = 370
ystop4 = 490
scale4 = 2

for i in range(8):
    heat = np.zeros_like(test_images[i][:,:,0]).astype(np.float)
    out_img, window_list1 = find_cars(test_images[i], ystart1, ystop1, scale1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    out_img, window_list2 = find_cars(test_images[i], ystart2, ystop2, scale2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    out_img, window_list3 = find_cars(test_images[i], ystart3, ystop3, scale3, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    out_img, window_list4 = find_cars(test_images[i], ystart4, ystop4, scale4, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    #print(window_list)
    
    window_list = window_list1 + window_list2 + window_list3 + window_list4
    #window_list = window_list1
    #window_list = window_list1 + window_list2 
    
    #Add heat to each box in box list
    heat = add_heat(heat, window_list)
    
    #Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    
    #Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    
    #Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(test_images[i]), labels)
    
    #Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 9))
    f.tight_layout()
    ax1.imshow(test_images[i])
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(labels[0], cmap='hot')
    ax2.set_title('Heat Map Image.', fontsize=15)
    ax3.imshow(draw_img, cmap='gray')
    ax3.set_title('Image with Car Positions', fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
```

#### 4. Test images

Apart from the 6 test images that are part of the project i also added 2 other images which have straight lines. I had used these 2 images in my previous project as well to gauge how well the intermediate steps were doing.

I will depict how these 8 images give a heat map image and an output with teh resulting bounding boxes.


![alt text][image1]


![alt text][image2]


![alt text][image3]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

