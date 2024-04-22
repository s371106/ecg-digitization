import sys
import scipy
#import cv2 as cv
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import pandas as pd


import cv2
import os
# import numpy as np
#from matplotlib import pyplot as plt
#from scipy import ndimage
#from PIL import Image
from matplotlib.pyplot import figure
from pathlib import Path
#from matplotlib import pyplot as plt
#from matplotlib.pyplot import figure
#import numpy as np
#import cv2
import math
#from scipy import ndimage
#import os
#import cv2


#original_image = cv2.imread("EKG_ACE2_20210511110737_page_0005.tif")
#original_image = cv2.imread(file_path)

project_directory = Path(__file__).parent
data_file_names = os.listdir(os.path.join(project_directory, "data"))
data_files_path = os.path.join(project_directory, "data")

for data_file_name in data_file_names:
    folder_name = data_file_name.replace(".tif", "")
    folder_path = os.path.join(project_directory, "output-digital", folder_name)

    try:
        os.mkdir(folder_path)
    except OSError:
        pass # Stopping termination when folder exists

    # Specify file path
    file_path = os.path.join(data_files_path, data_file_name)

    # Attempt to load image
    original_image = cv2.imread(file_path)

    # Check if image was loaded successfully
    if original_image is None:
        print(f"Error: Unable to load image from path: {file_path}")
    else:
        print("Image loaded successfully")

    # Continue with image processing...


    original_image = cv2.imread(file_path)
imag = cv2.resize(original_image, (0, 0), fx=0.5, fy=0.5)
image = imag.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred, 120, 255, 1)

# Find contours in the image
cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Obtain area for each contour
contour_sizes = [(cv2.contourArea(contour), contour) for contour in cnts]

# Find maximum contour and crop for ROI section
if len(contour_sizes) > 0:
    largest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    
    x,y,w,h = cv2.boundingRect(largest_contour)
    start_point = (x, y) 
    end_point = (x + w, y + h) 
  
    # Blue color in BGR 
    color = (255, 0, 0) 
  
    # Line thickness of 2 px 
    thickness = 2
  
    # Using cv2.rectangle() method 
    # Draw a rectangle with blue line borders of thickness of 2 px 
#     image = cv2.rectangle(image, start_point, end_point, color, thickness) 
    # area to be cropped
    ROI = image[y:y+h, x:x+w]

# cv2.imshow("canny", canny) 
# cv2.imshow("detected", image) 
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(ROI)
figure(figsize=(12, 8), dpi=80)

cv2.imwrite('cropped.png', ROI)

# erosion and dilation of images.
img = cv2.imread('cropped.png', 0)

# Taking a matrix of size 2 as the kernel
kernel = np.ones((2,2), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imwrite('Erosion.png', img_erosion)

mask = cv2.imread('Erosion.png', 0)

sliding_window_size_x = 9
sliding_window_size_y = 9

mean_filter_kernel = np.ones((sliding_window_size_x,sliding_window_size_y),np.float32)/(sliding_window_size_x*sliding_window_size_y)
filtered_image = cv2.filter2D(mask,-1,mean_filter_kernel)

invert = cv2.bitwise_not(filtered_image)
cv2.imwrite('filtered_image.png', invert)

image = cv2.imread('filtered_image.png')

# Gaussian Blur
Gaussian = cv2.GaussianBlur(image, (15, 15), 0)

light_orange = (0, 0, 0)
dark_orange = (120, 150, 150)
mask = cv2.inRange(Gaussian, light_orange, dark_orange)
invert = cv2.bitwise_not(mask)
cv2.imwrite('mask.png', invert)

def main():
    image_path = 'mask.png'
#     save_path = 'output_images/output' + image_path[-6:-4] + '.png'
    image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    print('INPUT image is of size: {} x {}.'.format(height, width))
    image = image[50 : height - 120, 50 : width - 20]
    ret, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], np.uint8)
    labeled_image, cc_num = ndimage.label(image, structure=structure)
    cc = ndimage.find_objects(labeled_image)
    print('There are {} connected components.'.format(cc_num))
    cc_areas = ndimage.sum(image, labeled_image, range(cc_num + 1))
    area_mask = cc_areas < 510000
    labeled_image[area_mask[labeled_image]] = 0
    labeled_image = np.where(labeled_image == 0, 255, 0)
    cv2.imwrite('try1.png',labeled_image)
    print('Image is saved ')

if __name__ == '__main__':
    main()

    img = cv2.imread('try1.png', 0)
kernel = np.ones((5,5), np.uint8)

img_erosion = cv2.erode(img, kernel, iterations=1)
invert1 = cv2.bitwise_not(img_erosion)
cv2.imwrite('try2.png',invert1 )

def main():
    image_path = 'try2.png'
#     save_path = 'output_images/output' + image_path[-6:-4] + '.png'
    image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    print('INPUT image is of size: {} x {}.'.format(height, width))
    image = image[0 : height - 0, 0 : width - 0]
    ret, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], np.uint8)
    labeled_image, cc_num = ndimage.label(image, structure=structure)
    cc = ndimage.find_objects(labeled_image)
    print('There are {} connected components.'.format(cc_num))
    cc_areas = ndimage.sum(image, labeled_image, range(cc_num + 1))
    area_mask = cc_areas < 860000
    labeled_image[area_mask[labeled_image]] = 0
    labeled_image = np.where(labeled_image == 0, 255, 0)
    cv2.imwrite('try3.png',labeled_image)
    print('Image is saved ')

if __name__ == '__main__':
    main()


image = cv2.imread('try3.png')

Gaussian = cv2.GaussianBlur(image, (19, 19), 0)

light_orange = (0, 0, 0)
dark_orange = (180, 150, 150)
mask = cv2.inRange(Gaussian, light_orange, dark_orange)
invert = cv2.bitwise_not(mask)
cv2.imwrite('try4.png', invert)

# Reading the input image
img = cv2.imread('try4.png', 0)
kernel = np.ones((11,11), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
cv2.imwrite('try5.png', img_erosion)

img = cv2.imread('try5.png', 0)
kernel = np.ones((10,10), np.uint8)
img_dilation = cv2.dilate(img, kernel, iterations=2)
invert = cv2.bitwise_not(img_dilation)
cv2.imwrite('try6.png', invert)





# Helper function to help display an oversized image
def display_image(image, name):
    if image.shape[0] > 1000:
        image = cv2.resize(image, (0, 0), fx=0.85, fy=0.85)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Helper function to sharpen the image
def sharpen(img):
    kernel = np.array([[0, -1, 0],
                        [-1, 5.5, -1],
                        [0, -1, 0]], np.float32)
    img = cv2.filter2D(img, -1, kernel)
    return img


# Helper function to increase contrast of an image
def increase_contrast(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


# Helper function to crop the image and eliminate the borders
def crop_image(image, upper, lower, left, right):
    mask = image > 0
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    image = image[x0 + upper: x1 + lower, y0 + left: y1 + right]
    return image


# Another helper function to crop and remove the borders
def crop_image_v2(image, tolerance=0):
    mask = image > tolerance
    image = image[np.ix_(mask.any(1), mask.any(0))]
    return image


# Helper function to distinguish different ECG signals on specific image
def separate_components(image):
    ret, labels = cv2.connectedComponents(image, connectivity=18)

    # mapping component labels to hue value
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_image = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_HSV2BGR)

    # set background label to white
    labeled_image[label_hue == 0] = 255
    return labeled_image


# Helper function to display segmented ECG picture
def display_segments(name, item, axis='off'):
    plt.figure(figsize=(12, 9))
    plt.imshow(item)
    plt.title(name)
    plt.axis(axis)
    plt.subplots_adjust(wspace=.05, left=.01, bottom=.01, right=.99, top=.9)
    plt.show()


# Helper function to detect characters
def ocr(image):
    text = pytesseract.image_to_string(image, lang='eng')
    return text


def main():
    image_name = 'try6.png'  # select image
#     image_name = 'new/6/22 陆金明 ？/IMAGE_000018.jpg'

    image = cv2.imread(image_name, flags=cv2.IMREAD_GRAYSCALE)  # read the image as GS

    # sanity check
    if image is None:
        print('Cannot open image: ' + image_name)
        sys.exit(0)

    display_image(image, 'Original Image')
    print(image.shape)

    # crop out upper region
#     cropped_image = crop_image(image, 0, 0, 0, 0)
#     display_image(cropped_image, 'Cropped Image')

    # use thresholding to transform the image into a binary one
#     ret, binary_image = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)
    binary_image = image
    display_image(binary_image, 'Binary Image')
    print(binary_image.shape)

    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], np.uint8)
    labeled_image, nb = ndimage.label(binary_image, structure=structure)
    display_segments('Labeled Image', labeled_image)

    print()
    print('There are ' + str(np.amax(labeled_image) + 1) + ' labeled components.')
    print()

    curve_indices = []
    curve_lengths = []
    curve_widths = []
    curve_lower_bound = []
    curve_upper_bound = []
    fig = plt.figure(figsize=(12, 8))
    plt.title('Separated Curves')
    columns = 1
    rows = 6
    for i in range(1, np.amax(labeled_image) + 1):
        sl = ndimage.find_objects(labeled_image == i)
        img = binary_image[sl[0]]
        if img.shape[1] > 100:
            curve_indices.append(i)
            curve_widths.append(img.shape[0])
            curve_lengths.append(img.shape[1])
            curve_lower_bound.append(sl[0][0].stop)
            curve_upper_bound.append(sl[0][0].start)
            print("Curve {} line range = [{}, {}].".format(len(curve_indices), sl[0][0].start, sl[0][0].stop))
            fig.add_subplot(rows, columns, len(curve_indices))
            plt.imshow(img, cmap='gray')
        else:
            continue
    plt.show()

    print("Effective curves are components from indices: ", curve_indices)
    print("Their corresponding curve lengths are: ", curve_lengths)
    print("Their corresponding curve widths are: ", curve_widths)
    print()

    fig = plt.figure(figsize=(12, 8))
    plt.title("Extracted 'S'")
    columns = 5
    rows = 2

    # for recording the baselines of the curves
    baselines = []
    for i in range(1, np.amax(labeled_image) + 1):
        sl = ndimage.find_objects(labeled_image == i)
        img = binary_image[sl[0]]
        if 10 < img.shape[0] < 12 and 6 < img.shape[1] < 8:
            if (len(baselines) == 6):
                break
            baselines.append(sl[0][0].start)
            print("'S' {} line range = [{}, {}].".format(len(baselines), sl[0][0].start, sl[0][0].stop))

            fig.add_subplot(rows, columns, len(baselines))
            plt.imshow(img, cmap='gray')
        else:
            continue
#     plt.show()

    print("The corresponding baselines for the curves are: ", baselines)
    print()

    fig = plt.figure(figsize=(12, 8))
    plt.title("Trimmed Curves")
    columns = 1
    rows = 6
    # make sure the curves have the same length (same as the shortest)
    final_images = []
    min_length = min(curve_lengths)
    for i in range(len(curve_indices)):
        sl = ndimage.find_objects(labeled_image == curve_indices[i])
        img = binary_image[sl[0]]
        # print(img.shape)
        if img.shape[1] > min_length:
            diff = img.shape[1] - min_length
            img = crop_image(img, 0, 0, diff, 0)
        final_images.append(img)
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img, cmap='gray')
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    plt.title('Scattered Dots')
    columns = 1
    rows = 6
  


    coords = []
    datafile = []
    for i in range(len(curve_indices)):
        curve = final_images[i]
        length = curve.shape[1]
        width = curve.shape[0]
        xs = []
        ys = []
        for j in range(length):
            for k in range(width - 1, -1, -1):
                if curve[k][j] == 255:
                    xs.append(j)
                    ys.append(width - k)
                    break
                else:
                    continue
        fig.add_subplot(rows, columns, i + 1)
        coords.append(ys)
        plt.plot(xs, ys)
#         print(ys)
        dict1  ={'x_val':xs, 'y_val':ys}
        datafile.append(dict1)
    df = pd.DataFrame(datafile[0])
    for dd in datafile[1:]:
        df = pd.concat([df, pd.DataFrame(dd)], axis=1)
    
    df.to_csv('test2.csv')
    
        
    bigger_pic = []
    for i in range(len(baselines)):
        axis = baselines[i]
        gs_img = []
        for j in range(len(coords[0])):
            actual_coord = curve_upper_bound[i] + coords[i][j]
            g = 127 + actual_coord - axis
            gs_img.append(g)
        bigger_pic.append(gs_img)
    array = np.array(bigger_pic, dtype=np.uint8)
    newimg = Image.fromarray(array)
    newimg.show()
    newimg.save('result_image.png')

if __name__ == '__main__':
    main()