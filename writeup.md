## Writeup

### Renyuan Zhang

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_calibration1.jpg "Undistorted"
[image2]: ./output_images/undistort_test1.jpg "Road Transformed"
[image3]: ./output_images/binary_test1.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/detect_lane_test1.jpg "Fit Visual"
[image6]: ./output_images/detect_lane_test2.jpg "Output"
[video1]: ./output_videos/detect_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in "./main.py" function `undistort_image()`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image using function generate_binary_image().  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_images()`, which appears in the file `main.py` (./output_images/warped_straight_lines1.jpg).  The `warp_images()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
        self.srcPoints = np.float32(
                        [[img_size[0] * 0.151, img_size[1]],
                        [img_size[0] * 0.451, img_size[1] * 0.64],
                        [img_size[0] * 0.553, img_size[1] * 0.64],
                        [img_size[0] * 0.888, img_size[1]]])

        self.dstPoints = np.float32(
                        [[(img_size[0] / 4) - 30, img_size[1]],
                        [(img_size[0] / 4 - 30), 0],
                        [(img_size[0] * 3 / 4 + 30), 0],
                        [(img_size[0] * 3 / 4 + 30), img_size[1]]])
```

This resulted in the following source and destination points:

```
print(a.srcPoints)
[[  193.27999878   720.        ]
 [  577.2800293    460.79998779]
 [  707.84002686   460.79998779]
 [ 1136.64001465   720.        ]]

print(a.dstPoints)
[[ 290.  720.]
 [ 290.    0.]
 [ 990.    0.]
 [ 990.  720.]]

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 140 through 146 in my code in `line.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `lanedetector.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/detect_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Likely to fail at different exposures. I need to think about how to adjust the image with different exposures. I also thinking about the driving in fog/raining days. How to adjust the contrast is also a problem.  
