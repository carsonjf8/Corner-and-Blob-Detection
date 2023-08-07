"""
Usage example:
python cforsyt_hw2_part2_code.py --window_size 5 --alpha 0.04 --corner_threshold 1000000000 hw3_images/house1.jpg
"""

import cv2
import numpy as np
import sys
import getopt
import operator
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpim

def findCorners(img, window_size, k, thresh):
    """
    Finds and returns list of corners and new image with corners drawn
    :param img: The original image
    :param window_size: The size (side length) of the sliding window
    :param k: Harris corner constant. Usually 0.04 - 0.06
    :param thresh: The threshold above which a corner is counted
    :return:
    """
    #Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    plt.imshow(dx, cmap='gray')
    plt.show()
    plt.imshow(dy, cmap='gray')
    plt.show()
    mag = (Ixx + Iyy)**0.5
    plt.imshow(mag, cmap='gray')
    plt.show()
    orien = np.arctan(dy, dx)
    orien += np.pi
    orien /= 2 * np.pi
    plt.imshow(orien, cmap='gray')
    plt.show()

    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = math.floor(window_size/2)
    
    # Loop through image and find our corners
    # and do non-maximum supression
    # this can be also implemented without loop

    print("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            '''
            given second moment matrix instead of computing eigenvalues 
            and eigenvectors you can compute the corner response funtion 
            using trace and det
            Find determinant and trace, use to get corner response

            det = (Mxx * Myy) - (Mxy**2)
            trace = Mxx + Myy
            r = det - k*(trace**2)
            '''

            Mxx = np.sum(Ixx[y - offset: y + offset + 1, x - offset: x + offset + 1])
            Myy = np.sum(Iyy[y - offset: y + offset + 1, x - offset: x + offset + 1])
            Mxy = np.sum(Ixy[y - offset: y + offset + 1, x - offset: x + offset + 1])

            det = (Mxx * Myy) - (Mxy ** 2)
            trace = Mxx + Myy
            r = det - k * (trace ** 2)

            if r > thresh:
                cornerList.append([x, y, r])

    cornerList = nms(cornerList)
    for corner in cornerList:
        x = corner[0]
        y = corner[1]

        size = [-2, -1, 0, 1, 2]
        for i in size:
            for j in size:
                color_img.itemset((y + i, x + j, 0), 0)
                color_img.itemset((y + i, x + j, 1), 0)
                color_img.itemset((y + i, x + j, 2), 255)
    return color_img, cornerList

def nms(cl):
    too_close = 5
    cl = sorted(cl, key=lambda x: x[2], reverse=True)
    cl_new = []
    for c in cl:
        overlap = False
        for c_n in cl_new:
            if dist(c, c_n) < too_close:
                overlap = True
                break
        if not overlap:
            cl_new.append(c)
    return cl_new

def dist(pt1, pt2):
    return math.sqrt(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2))

def main():
    """
    Main parses argument list and runs findCorners() on the image
    :return: None
    """
    args, img_name = getopt.getopt(sys.argv[1:], '', ['window_size=', 'alpha=', 'corner_threshold='])
    args = dict(args)
    print(args)
    window_size = args.get('--window_size')
    k = args.get('--alpha')
    thresh = args.get('--corner_threshold')

    print("Image Name: " + str(img_name[0]))
    print("Window Size: " + str(window_size))
    print("K alpha: " + str(k))
    print("Corner Response Threshold:" + thresh)

    img = cv2.imread(img_name[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    finalImg, cornerList = findCorners(img, int(window_size), float(k), int(thresh))
        
    points = np.array(cornerList) 
    plot = plt.figure(1)
    plt.imshow(img, cmap="gray")
    plt.plot(points[:,0],points[:,1], 'b.')
    plt.show()
         
    if finalImg is not None:
        cv2.imwrite("./hw3_sample_output/corners_house1.png", finalImg)

if __name__ == "__main__":
    main()
    