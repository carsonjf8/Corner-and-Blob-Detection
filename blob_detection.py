"""
Usage example:
python cforsyt_hw2_part2_code.py -s -d hw3_images/butterfly.jpg
-d : downsampling
-f : changing filter size
-s : display image
"""

import cv2
import numpy as np
import sys
import getopt
import operator
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import scipy
import skimage
import time

def findBlobsDownsample(img, filename, show_img):
    # img is already grayscale
    # convert img vals from 0-255 to 0.0-1.0
    img = img / 255
    xyrv = []
    thresh = 0.08
    scales_count = 9
    multiplier = 2
    sigma = 2

    scale_space = np.empty(scales_count, dtype=object)

    # iterate through each scale
    for i in range(scales_count):
        print('Scale', i)
        multiplier_power = pow(multiplier, i)
        resized_width = int(img.shape[1] / multiplier_power)
        resized_height = int(img.shape[0] / multiplier_power)
        # resize image
        resized_img = skimage.transform.resize(img, (resized_height, resized_width))

        # apply LoG filter
        print('Applying LoG filter...')
        log_img = scipy.ndimage.filters.gaussian_laplace(resized_img, sigma=sigma)
        log_img *= pow(sigma, 2)
        log_img = np.abs(log_img)

        resized_img_color = np.empty((resized_img.shape[0], resized_img.shape[1], 3))
        resized_img_color[:, :, 0] = log_img
        resized_img_color[:, :, 1] = log_img
        resized_img_color[:, :, 2] = log_img
        print('Thresholding...')
        for j in range(log_img.shape[0]):
            for k in range(log_img.shape[1]):
                val = log_img[j, k]
                if val > thresh:
                    xyrv.append([k * multiplier_power, j * multiplier_power, np.sqrt(2) * sigma * multiplier_power, val])

        scale_space[i] = log_img

    print('Running NMS...')
    xyrv = blob_nms(xyrv)
    xyrv = np.array(xyrv)

    print('Displaying blobs')
    show_all_circles(img, xyrv[:, 0], xyrv[:, 1], xyrv[:, 2], filename, 'blobs_downsample_', show_img)

def findBlobsFilter(img, filename, show_img):
    # img is already grayscale
    # convert img vals from 0-255 to 0.0-1.0
    img = img / 255
    xyrv = []
    thresh = 0.08
    scales_count = 9
    multiplier = 2
    initial_sigma = 2

    scale_space = np.empty(scales_count, dtype=object)

    # iterate through each scale
    for i in range(scales_count):
        sigma = initial_sigma * (i + 1)

        # apply LoG filter
        print('Applying LoG filter...')
        log_img = scipy.ndimage.filters.gaussian_laplace(img, sigma=sigma)
        log_img *= pow(sigma, 2)
        log_img = np.abs(log_img)

        print('Thresholding...')
        for j in range(log_img.shape[0]):
            for k in range(log_img.shape[1]):
                val = log_img[j, k]
                if val > thresh:
                    xyrv.append([k, j, np.sqrt(2) * sigma, val])

        scale_space[i] = log_img

    print('Running NMS...')
    xyrv = blob_nms(xyrv)
    xyrv = np.array(xyrv)

    print('Displaying blobs')
    show_all_circles(img, xyrv[:, 0], xyrv[:, 1], xyrv[:, 2], filename, 'blobs_filter_', show_img)

def blob_nms(bl, too_close=10):
    bl = sorted(bl, key=lambda x: x[3], reverse=True)
    bl_new = []
    for b in bl:
        overlap = False
        for b_n in bl_new:
            if dist(b, b_n) < too_close:
                overlap = True
                break
        if not overlap:
            bl_new.append(b)
    return bl_new

def dist(pt1, pt2):
    distance = math.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))
    return distance

def show_all_circles(image, cx, cy, rad, filename, fn_prefix, show_img, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.savefig('./hw3_sample_output/' + fn_prefix + filename)
    if show_img:
        plt.show()

def main():
    opts, args = getopt.getopt(sys.argv[1:], 'dfs')

    print("Image Name: " + str(args[0]))

    path = args[0]
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filename = path.split('/')[-1]
    filename = filename.split('\\')[-1]

    show_img = False
    for opt, arg in opts:
        if opt in ['-s']:
            show_img = True

    for opt, arg in opts:
        if opt in ['-d']:
            start = time.time()
            findBlobsDownsample(img, filename, show_img)
            end = time.time()
            print(end - start)
        elif opt in ['-f']:
            start = time.time()
            findBlobsFilter(img, filename, show_img)
            end = time.time()
            print(end - start)
        else:
            print('Unhandled option', opt)

if __name__ == "__main__":
    main()
    