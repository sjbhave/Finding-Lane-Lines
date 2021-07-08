# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


---

### Reflection

My pipeline consist of 6 step 

1. Converting image into grascale imaage and appliying Gaussian filter for smoothing the image 

Image is made of pixels a 3 channel color image, Each pixel is combinarion of 3 intensity value red, green and blue channel 

where as grayscale image only one channel, each pixel with only one intensity value 

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

Image noise can create false edge detection hence need to smmoth it 

Gaussian Filter: 

Image is stored as a collection of discrete pixels, each pixel of the gray scale image is represented by single number that describe the brightness 

For smoothing we modify the vale of pixel with avarge value of the pixel intensity, Avaraging out pixel in the image to reduce the noise will be done with the kernal 

Kernal of normally distributed number is run across our entire image and sets each pixel values equal to the weighted avarage of it's neighbouring pixel, thus smoothing the image 

blur = cv2.GaussianBlur(gray,(5,5),0)

I am applying a gaussian blur on a grayscale image with a five by five kernal 

Simple Edge Detection:

We can look our image as an array but also a continous function of X and Y 

X- Number of columns in the image, Y- Number of Rows in the image 

product of x and Y gives the total number of pixel in your image 

The canny function performs the derivative both in x and Y directions thereby measuring the change in intensity with respect to adjusant pixels 

It computes the gradient in all the directions of our blurred image and then going to trace our strongest gradient as series of white pixel 

canny = cv2.Canny(blur,50,150)

Low threshold and high threshold actully allows to isolate pixel that follows the strongest gradient 

2. Finding Region of Intrest 

From the Dimention of image I have masked the image first in the polygon 

height = image.shape[0]
polygons = np.array([[(100,height),(870,height),(500,300)]])

that perticular polygon is converted to white image 
mask = np.zeros_like(image)
white_image = cv2.fillPoly(mask,polygons,255)

using bitwiseand function found the lane lines in the region 

masked_image = cv2.bitwise_and(image,white_image)

3. Detecting the line using Hough transform 

In the Hough transform, the points or lines are represented in polor coordinates rho and theta

so point is represented by curve and line is represented by point 

now if in the x and Y axis if there is locus of points, through which line can be drawn, these points and line can be represented in the rho and theta graph as number of curve intercepting, more the intercepting curve at, more points the locus is covering 

lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,20,np.array([]),minLineLength=5,maxLineGap = 200)

the abouve equation find the lines array , here threshold = 100 that is minimum number of intersection nedded to accept as line 

minLineLength is mimimum lenghth of line to accept as line, I kept it 5

MaxLineGap = maxmum gap so that it can diiferentaite two line, I kept it 200

rho = 2 distance resolution, theta = pi/180 is the angle resolution that is 1 degree 


4. Optimizing 

The output of lines is array of lines which follows above criteria

In this there are some lines which represent left lane and some lines which represents right lane 

so first we have calculated slope and interept of all the lines

parameters = np.polyfit((x1,x2),(y1,y2),1)
slope = parameters[0]
intercept = parameters[1]

and if slope is smaller than 0 then it is left_fit and if slope is greater than 0 then it is right_fit 

after that avarage of left_fit slope, intercept and avarage of right_fit slope and intercept is calculated 

and from the slope and intecept left_line coordinates and right line coordinates are calulated 

def avarage_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        #print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average
    return np.array([left_line,right_line])
    
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])
    
4. Converting lines into line image  

Line is in the form of [x1,y1,x2,y2] , we are converting it into line image, value of only 1 channel we have given hence it looking as Red 

(255,0,0) thickness is decided as 8

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            #print(line)
            x1,y1,x2,y2 = line.reshape(4) 
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),8)
    return line_image 
    
5. Combination of orignal image and line image 

Finally combination of orignal image and line_image is found 

line_image = display_lines(lane_image,avaraged_lines)
combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)

6. processing on the Vedio 

Test Video is taken as input, Vedeo is sequence of images
same image processing is done on the each frame of the vedio
with specific time of 0.01sec
cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap = 5)
    avaraged_lines = avarage_slope_intercept(frame,lines)
    line_image = display_lines(frame,avaraged_lines)
    combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
