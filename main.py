### Spliting a single cropped scan into six overlapping strips and then applying and testing different CCL threshold values for each strip
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import cv2
import math
from scipy import ndimage
import os
import cv2
from pathlib import Path

# Command to transfer files in a folder: scp -r <folder> ubuntu@10.196.38.72:/home/ubuntu/New/data

# Load the original image
# project folder
# - main.py
# - main.py
# - data
#   -  EKG_ACE2_20210511110737_page_0010.tif
#   -  EKG_ACE2_20210511110737_page_0010.tif
#   -  EKG_ACE2_20210511110737_page_0010.tif
#   -  EKG_ACE2_20210511110737_page_0010.tif
#   -  ...
# - venv (perhaps needed)



project_directory = Path(__file__).parent
data_file_names = os.listdir(os.path.join(project_directory, "data"))
data_files_path = os.path.join(project_directory, "data")

for data_file_name in data_file_names:
    folder_name = data_file_name.replace(".tif", "")
    folder_path = os.path.join(project_directory, "output", folder_name)

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


    original_image = cv2.imread(file_path) #python/python3 ./main.py

    #original_image = cv2.imread(r"D:\ACIT - Oslomet\9. Master Thesis\Ahus Scanns\EKG_ACE2_20210511110737_page_0010.tif")

    ### Cropping
    imag = cv2.resize(original_image, (0, 0), fx=0.4, fy=0.4)
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

        x, y, w, h = cv2.boundingRect(largest_contour)

        ROI = image[y:y + h, x:x + w]

    # path = "D:\ACIT - Oslomet\9. Master Thesis\Stored_images"
    cv2.imwrite(os.path.join(folder_path, 'cropped.png'), ROI)

    ### Rotating
    img_before = cv2.imread(os.path.join(folder_path, 'cropped.png'))
    img_before_copy = img_before.copy()
    img_gray = cv2.cvtColor(img_before_copy, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 5, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img_before_copy, (x1, y1), (x2, y2), (255, 0, 0), 1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)

    rotated = ndimage.rotate(img_before, median_angle, cval=255)
    cv2.imwrite(os.path.join(folder_path, 'rotated.jpg'), rotated)

    ### Erosion
    img = cv2.imread(os.path.join(folder_path, 'rotated.jpg'), 0)

    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite(os.path.join(folder_path, 'erosion.jpg'), img_erosion)

    ### Blurring
    mask = cv2.imread(os.path.join(folder_path, 'erosion.jpg'), 0)

    sliding_window_size_x = 9
    sliding_window_size_y = 9

    mean_filter_kernel = np.ones((sliding_window_size_x, sliding_window_size_y), np.float32) / (
                sliding_window_size_x * sliding_window_size_y)
    filtered_image = cv2.filter2D(mask, -1, mean_filter_kernel)

    invert = cv2.bitwise_not(filtered_image)
    cv2.imwrite(os.path.join(folder_path, 'filtered_image.png'), invert)

    ### Spliting the scan into six strips
    before_spliting = cv2.imread(os.path.join(folder_path, 'filtered_image.png'))
    height = before_spliting.shape[0]
    width = before_spliting.shape[1]
    cropped_50 = before_spliting[50: height - 50, 50: width - 50]
    slice1 = cropped_50[:370, :]
    slice2 = cropped_50[160:590, :]
    slice3 = cropped_50[390:820, :]
    slice4 = cropped_50[620:1050, :]
    slice5 = cropped_50[850:1280, :]
    slice6 = cropped_50[1080:, :]

    slice_list = [slice1, slice2, slice3, slice4, slice5, slice6]

    # making a list with different CCL threshold values
    cc_area_list = []
    for i in range(10):
        y = 30000 * i
        cc_area_list.append(y)

    structure = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.uint8)

    ### Unmark the followings in case you need to apply erosion and blurring in every iteration of the loop
    # kernel_loop = np.ones((2,2), np.uint8)
    # sliding_window_x_loop = 9
    # sliding_window_y_loop = 9
    # mean_filter_kernel_loop = np.ones((sliding_window_x_loop,sliding_window_y_loop),np.float32)/(sliding_window_x_loop*sliding_window_y_loop)


    ### Applying different threshold values for every strip and save the outcome
    for j in range(6):
        image = "slice" + str(j + 1) + ".png"
        cv2.imwrite(os.path.join(folder_path, image), slice_list[j])
        strip = cv2.imread(os.path.join(folder_path, image))
        for i in range(10):
            dark = (0, 0, 0)
            light = (120, 150, 150)
            mask = cv2.inRange(strip, dark, light)
            invert = cv2.bitwise_not(mask)
            labeled_image, cc_num = ndimage.label(invert, structure=structure)
            cc_areas = ndimage.sum(invert, labeled_image, range(cc_num + 1))
            area_mask = cc_areas < cc_area_list[i]
            labeled_image[area_mask[labeled_image]] = 0
            labeled_image = np.where(labeled_image == 0, 255, 0)

            cv2.imwrite(os.path.join(folder_path, 'temp.png'), labeled_image)

            outcome_name = "slice" + str(j + 1) + "_" + str(cc_area_list[i]) + ".png"
            cv2.imwrite(os.path.join(folder_path, outcome_name), labeled_image)

    # Unmark the followings in case you need to apply erosion and blurring in every iteration of the loop
    #         outcome = cv2.imread(os.path.join(folder_path,'temp.png'))
    #         outcome_erosion = cv2.erode(outcome, kernel_loop, iterations=1)
    #         outcome_filtered = cv2.filter2D(outcome_erosion,-1,mean_filter_kernel_loop)
    #         strip=cv2.bitwise_not(outcome_filtered)
