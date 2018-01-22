## Self Driving Car Nanodegreee
### Vehicle Detection and Tracking

---

[//]: # (Image References)
[image1]: ./project_images/combo3in1_a.jpg
[image2]: ./project_images/combo3in1_b.jpg
[image3]: ./project_images/combo3in1_b.jpg
[image4]: ./project_images/vehicle_hog_1.jpg
[image5]: ./project_images/vehicle_hog_2.jpg
[image6]: ./project_images/vehicle_hog_3.jpg
[image7]: ./project_images/vehicle_hog_4.jpg
[image8]: ./project_images/vehicle_hog_5.jpg
[image9]: ./project_images/non_vehicle_hog_1.jpg
[image10]: ./project_images/non_vehicle_hog_2.jpg
[image11]: ./project_images/non_vehicle_hog_3.jpg
[image12]: ./project_images/non_vehicle_hog_4.jpg
[image13]: ./project_images/non_vehicle_hog_5.jpg
[image14]: ./project_images/hog_sub_sampled_image_1.jpg
[image15]: ./project_images/hog_sub_sampled_image_2.jpg
[image16]: ./project_images/hog_sub_sampled_image_3.jpg
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

#### 1. Extraction of HOG features.

The first step of the project is to read in the various images for the vehicles and non vehicles.

For this I have a function that I essentially call twice; once to obtain vehicles and the second time to obtain non vehicles.

```sh
def get_images(path):
    images = []
    for dirs, subdir, files in os.walk(path):
        print(dirs, subdir)
        for file in files:
            if '.DS_Store' not in file:
                images.append(os.path.join(dirs, file))
                
    return list(map(lambda img: mpimg.imread(img), images))
```

To obtain vehicles I have the following snippet.

```sh
vehicle_path = '../vehicles'
vehicles = get_images(vehicle_path)
print(len(vehicles))
```

To obtain non vehicles I have the following snippet.


```sh
non_vehicle_path = '../non-vehicles'
non_vehicles = get_images(non_vehicle_path)
print(len(non_vehicles))
```

I then output 5 random car images along with their associated HOG images.

The images can be seen below after the code snippet.

```sh
for i in range(number_of_vehicle_images):
    index = random.randint(0, len(vehicles))
    image = vehicles[index]

    features, hog_image = get_hog_features(image[:,:,0], 9, 8, 2, True, True)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('Hog Image.', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
```

![alt text][image4]


![alt text][image5]


![alt text][image6]


![alt text][image7]


![alt text][image8]



I then output 5 random non car images along with their associated HOG images

```sh
for i in range(number_of_non_vehicle_images):
    index = random.randint(0, len(non_vehicles))
    image = non_vehicles[index]

    features, hog_image = get_hog_features(image[:,:,0], 9, 8, 2, True, True)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('Hog Image.', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

```

![alt text][image9]


![alt text][image10]


![alt text][image11]


![alt text][image12]


![alt text][image13]


#### 2. Choice of HOG parameters.

I tried various combinations of parameters and finally settled for the below.

```sh
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial = 16
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
```

I gauged how the output images with bounding boxes looked when there were cars in the images. There were cases when i couldn't have a bounding box associated with a car. This meant that i would not be able to detect the car. I also consulted the udacity forum and YCrCb color space seemed to be able to give good results so i settled for YCrCb color space along with the above mentioned parameters.

#### 3. SVM

We essentially have a classification problem here since we have to choose between **car** and **not car**. This is NOT a regression problem. Since we have a binary classification problem, a decision tree might also be able to produce good results. However, i have not tried this in my project. I intend to try this at a later date.

The code for the SVM is as below. I have re used the code from the project helper videos.
As can be seen below after defining the label vecor i split the data with a ratio of 0.2.
I use a linear SVC and then fit the model and finally make some predictions.

```sh
#Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


#Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of size',spatial,
    'and', hist_bins,'histogram bins')
print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
#Use a linear SVC 
svc = LinearSVC()
#Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
#Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
#Check the prediction time for a single sample
t=time.time()
n_predict = 20
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
```


**The results from the above code are below.**


```sh
Using spatial binning of size 16 and 32 histogram bins
Using: 8 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5568
16.64 Seconds to train SVC...
Test Accuracy of SVC =  0.9863
My SVC predicts:  [ 0.  1.  1.  1.  1.  1.  0.  0.  1.  0.  0.  0.  1.  0.  0.  1.  1.  0.
  1.  0.]
For these 20 labels:  [ 0.  1.  1.  1.  1.  1.  0.  0.  1.  0.  0.  0.  1.  0.  0.  1.  1.  0.
  1.  0.]
0.00397 Seconds to predict 20 labels with SVC

```

I should leverage GridSearchCV to obtain a better model.


### Sliding Window Search

#### 1. Finding Cars

For this part of the project, i decided to re use the find cars routine provided in the project helper videos. For the test images provides, once scale seems enough to detect the car. This is however, not a viable option for the video which has several frames. I found this out by experimenting.

The project videos explained the find cars function which needs to extract hog features only once and can be sub sampled to get all the the available overlay windows. These windows have an associated scale factor that define the overlap. To have different levels of overlap, the function needs to be called multiple times. This is what i did for the video. My pipeline to process individual frames of the video actually calls the find cars funtions four times. Care needs to be taken at this stage since cars that are father away will tend to be smallet and so the scale needs to be dealt with appropriatley.

I have a minor but important tweak to the find cars implementation that was provided by udacity. 
I have a list called window_list.

Every time i correctly predict a car, i add the co-ordinates to this list. I also return the list to the calling sequence.

```sh
window_list.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
```

While finding cars, one of the inputs to the function happen to be the svc model. So it is important to have a good model. Again based on experimentation, the larger the feature vector length the better the model accuracy.

#### 2 Optimizing Classifier

I initially started of with RGB as the color space and even experimented with HSV as i seemed to have trouble classifying the white car. I settled for YCrCb since it gave me the best results.

```sh
hog_channel = 'ALL'
```

I also make the above selection for the hog channel since i don't want to loose any features and this also increases my feature vector length.

#### 3. Initial Pipeline on Test Images

For the test images, i used one scale. I provide the details below. Once, i observed the output images, i could gauge that i had atleast a fucntional pipeline and i had correctly integrated the various components of the project.

For this part of the project, the main pieces of code are as follows. I used a scale of **1.5**

**PLEASE NOTE** : It became obvious when I went onto test on the videos that i needed to have more scales. In total I have for the video section of the project 3 different search areas with 3 scales. I explain this in the section below.


```sh

ystart1 = 336
ystop1 = 656
scale1 = 1.5

for i in range(8):
    out_img, window_list1 = find_cars(test_images[i], 336, 656, 640, 1280, scale1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    #Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    f.tight_layout()
    ax1.imshow(test_images[i])
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(out_img, cmap='gray')
    ax2.set_title('Hog Sub Sampled Image.', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

```


![alt text][image14]


![alt text][image15]


![alt text][image16]


---

### Video Implementation

#### 1. Image Pipeline


I have four different scales and search areas that i settled for after much experimentation. One of the main difficulties in this part of the project was detection of the **white car**. As this was moving away, the effective size/scale of the **car image** was changing so i had to make changes to search area and scale. The end result after this is not perfect but atleast detects the **white car** most of the time.

So after obtaining 3 window lists from the 3 different calls to the find cars routine, I combine the three window lists into one list. I then add heat to each box in box list and apply a threshold of **two** and also **five** to help remove false positives. Finally i find final boxes from heatmap using label function. Essentially each blob corresponds to a vehicle and bounding boxes are constructed to cover the area of each detected blob. I have two separate outputs to gauge the difference when i add a threshold of **two** and **five**. I settled to use a threhold of five for the video pipeline since the output test images seem better with a threhold of five.


The code to do the above is represented below.

```sh
xstart = 640
xstop = 1280

ystart1 = 400
ystop1 = 656
scale1 = 1.4

ystart2 = 400
ystop2 = 592
scale2 = 1.6

ystart3 = 336
ystop3 = 528
scale3 = 1.9

for i in range(6):
    heat = np.zeros_like(test_images[i][:,:,0]).astype(np.float)
    out_img, window_list1 = find_cars(test_images[i], ystart1, ystop1, xstart, xstop, scale1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    out_img, window_list2 = find_cars(test_images[i], ystart2, ystop2, xstart, xstop, scale2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    out_img, window_list3 = find_cars(test_images[i], ystart3, ystop3, xstart, xstop, scale3, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    window_list = window_list1 + window_list2 + window_list3
    
    #Add heat to each box in box list
    heat = add_heat(heat, window_list)
    
    #Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    
    #Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    
    #Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(test_images[i]), labels)
    
    #Apply threshold to help remove false positives
    heat = apply_threshold(heat, 5)
    
    #Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    
    #Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img1 = draw_labeled_bboxes(np.copy(test_images[i]), labels)
    
    #Plot the result
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 9))
    f.tight_layout()
    ax1.imshow(test_images[i])
    ax1.set_title('Original Image', fontsize=12)
    ax2.imshow(labels[0], cmap='hot')
    ax2.set_title('Heat Map Image.', fontsize=12)
    ax3.imshow(draw_img, cmap='gray')
    ax3.set_title('Image with Heat Thresh 2', fontsize=12)
    ax4.imshow(draw_img1, cmap='gray')
    ax4.set_title('Image with Heat Thresh 5', fontsize=12)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
```


#### 2. Test images

I am testing using the 6 test images that have been provided.

Below, i depict 2 separate images with a heat threshold of 2 and 5. We can see that a threshold of 5 gives a slightly better output so i am going to continue using that in my project.


![alt text][image1]


![alt text][image2]



#### 3. Test Video

Here's a [link to my test video output][video1]

As we can see above, both cars are successfully detected.


#### 4. Project Video

Here's a [link to my project video output][video2]

In the last frame both black cars in the vicinity are detected.

---

### Discussion


- The biggest issue that i faced was since i was calling the Hog Sub Sampling Window Search with 4 different scales/points, the turn around time to get the actual output video even from an Amazon EC2 GPU instance was around 30 minutes. This caused even minor updates to take a long time to produce tangible results. Outputs from the test video were usually fine, but were not always reflective of the larger project video.

- I need to use Grid Search CV in my project to get a better model.

- I should also try other models such as Decision Tree Classifier.

- I notice that the windows are wobbly, a better sliding window search with either better or more scales will make the Pipeline more robust.

