# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import cv2, os, time, math, itertools, random
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from itertools import groupby
from operator import itemgetter
import line_equations
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
import pandas as pd
import tkinter.filedialog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import cv2, os, time, math, itertools, random
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.morphology import skeletonize
from sklearn.preprocessing import binarize
from skimage.morphology import binary_closing
from skimage.morphology import binary_dilation
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.filters import threshold_local
from itertools import groupby
from operator import itemgetter
import line_equations
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import warnings
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
import pandas as pd
import tkinter.filedialog
from skimage.measure import compare_ssim
import imutils
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
import csv

warnings.filterwarnings("ignore", category=DeprecationWarning)

cwd = os.getcwd()
path = os.path.join(cwd, 'training_mmse_pentagons')
model_path = os.path.join(path, "models", "svm.model")


def line_points(start, end):
    "Bresenham's line algorithm"
    x0, y0 = start
    x1, y1 = end
    points_in_line = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append((x, y))
    return points_in_line


def detection(img):
    '''
    The detections methond using the HOG algorithm
    '''

    clf = joblib.load(model_path)  # prediciton method using SVM
    # Binarize.
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # preprocessing in of image
    ret, thesh_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thesh_image, cv2.MORPH_OPEN, kernel)
    opening = cv2.resize(opening, (128, 128))

    fd, _ = hog(opening, 9, (8, 8), (3, 3), visualise=True,
                transform_sqrt=True)  # defined parameters for the HOG method
    return clf.decision_function([fd])


def find_contours(path, corners, quality, distance, detection_threshold):
    '''
    cv2 contour finding to get the shapes on page according to the black pixels
    '''
    found_dets = []
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    clone = img.copy()
    clean = img.copy()

    group_x = []
    group_y = []

    # img = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    # color_im = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # clone = color_im.copy()
    # thresh = threshold_otsu(img)
    # binary = img > thresh
    #
    inv_img = np.invert(im)
    binary = binarize(inv_img, threshold=15.0)
    # binary = np.invert(binary)
    skel = skeletonize(binary)
    # ##plt.imshow(skel)
    # ##plt.show()
    cv_skel = img_as_ubyte(skel)

    blur = cv2.GaussianBlur(im, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ##    cv2.imshow("thresh", th3)
    ##    cv2.waitKey()

    edges = cv2.Canny(img, 0, 255, apertureSize=3)  # Canny image
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # create the kernle
    # kernel = np.ones((5,5),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=5)
    # erosion = cv2.erode(im_reshape,kernel,iterations = 1)
    # opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    ##    cv2.imshow('Edges', edges)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # find each contour
    cv2.drawContours(clone, contours, -1, (0, 255, 0), 3)  # draw all contours
    # cv2.imshow("ALL CONT", clone)
    found_cnts = []
    rectangles = []
    features = []

    cv2.drawContours(edges, contours, -1, (0, 255, 255), 3)
    for cnt in contours:
        # get the perimeter circularity of the contours
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull,True)
        cv2.drawContours(img, cnt, -1, (0,255,255), 3)
        if perimeter != 0:
            x,y,w,h = cv2.boundingRect(cnt)
            if (h*w) > 3000: # minimum area of the contour bounding box
                feature = im[y:y + h, x:x + w]
                features.append(feature)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cd = detection(feature)
                if (cd > detection_threshold):
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    found_dets.append((x, y, cd, w, h))
    det_compares = itertools.combinations(found_dets,2)
    for i in det_compares: # Ignore the small detecction inside a bigger detection
        det1, det2 = i
        x1,y1, _, w1,h1 = det1
        x2,y2, _, w2,h2 = det2
        rect1_p1 = x1,y1
        rect1_p2 = x1+w1, y1+h1
        rect2_p1 = x2,y2
        rect2_p2 = x2+w2, y2+h2
        if ( rect1_p1[0] < rect2_p1[0] < rect2_p2[0] < rect1_p2[0] and rect1_p1[1] < rect2_p1[1] < rect2_p2[1] < rect1_p2[1]):
            small = min(i, key=lambda x: x[3]*x[4])
            if small in found_dets:
                found_dets.remove(small)
    for i in found_dets:
        x, y, _, w, h = i
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        croppedImage = clean[y:y+h, x:x+w]

    return croppedImage, img, found_dets


def detection_funct(path):
    detections, found_dets = corner_dets_methods.find_contours(path, corner_num, quality, distance, detection_threshold)
    return detections, found_dets


def corner_details(image, corner_image, line_image, detections):
    shi_c, shi_lines, distilled_lines = corner_dets_methods.cornerMeths(image,[detections],corner_num,quality,
                                                                                       distance,
                                                                                       line_threshold)
    line_count = len(shi_lines)
    # apply_corners(corner_image, shi_c)
    # apply_lines(line_image, shi_lines, distilled_lines)
    corner_count = len(shi_c)
    return corner_count, line_count


path = tkinter.filedialog.askopenfilename()
print(path)
cropped, im, found_dets = find_contours(path, 50, .2, 2, 0)

img = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
color_im = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
clone = color_im.copy()
##thresh = threshold_otsu(img)
##binary = img > thresh
inv_img = np.invert(img)
binary = binarize(inv_img, threshold=15.0)
#binary = np.invert(binary)
skel = skeletonize(binary)
##plt.imshow(skel)
##plt.show()
cv_skel = img_as_ubyte(skel)
lines = probabilistic_hough_line(skel,
                                 threshold=0,
                                 line_length=4,
                                 line_gap=10)


####### HOUGH EXAMPLE ################
from matplotlib import cm

h, theta, d = hough_line(skel)
fig, axes = plt.subplots(3, 4, figsize=(20, 11),
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(skel, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1 / 1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(skel, cmap=cm.gray)
line_count = 0
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - skel.shape[1] * np.cos(angle)) / np.sin(angle)
    # print(angle, dist)
    line_count += 1
    ax[2].plot((0, skel.shape[1]), (y0, y1), '-r')
ax[2].set_xlim((0, skel.shape[1]))
ax[2].set_ylim((skel.shape[0], 0))
ax[2].set_xlabel('Line count = ' + str(line_count))
# ax[2].set_axis_off()
ax[2].set_title('Hough Lines')

from matplotlib import collections  as mc

c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

lc = mc.LineCollection(lines, colors=c, linewidths=2)
ax[3].add_collection(lc)
ax[3].invert_yaxis()
ax[3].autoscale()
ax[3].margins(0.1)
ax[3].set_xlabel('Line count = ' + str(len(lines)))
ax[3].set_title('Probablistic Hough')

match_p = np.where(skel == 1)
##ax[4].scatter(match_p[1], match_p[0])
##ax[4].invert_yaxis()
##

im2, contours, hierarchy = cv2.findContours(cv_skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #
##cv2.drawContours(color_im, contours, -1, (0,0,255), 3)
##plt.imshow(color_im)
##plt.show()
all_cnt_p = []
x = match_p[0]
y = match_p[1]
black_coords = list(zip(x, y))
ax[5].scatter(y, x)
edit_black = []
# print(black_coords)
cnt_count = 0
ends = []
ext_points = []
for cnt in contours:
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(hull)
    # print(area)
    perimeter = cv2.arcLength(hull, True)
    if area > 25:
        # print("MAX  ", cnt)
        # determine the most extreme points along the contour
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
        ext = [extLeft, extRight, extTop, extBot]
        ext_points.append(ext)

        # print("L: ", extLeft, "R: ", extRight, "T: ", extTop, "B: ", extBot)
        cv2.circle(color_im, extLeft, 6, (0, 0, 255), -1)
        cv2.circle(color_im, extRight, 6, (0, 255, 0), -1)
        cv2.circle(color_im, extTop, 6, (255, 0, 0), -1)
        cv2.circle(color_im, extBot, 6, (255, 255, 0), -1)

        cnts_x = []
        cnts_y = []
        # print("CNT Count = ", cnt_count)
        color = np.random.uniform(0, 255, 3)
        cv2.drawContours(color_im, cnt, -1, color, 3)
        cnt_points = []
        # print(len(black_coords))
        for i in black_coords:
            # print(i)
            inside = cv2.pointPolygonTest(cnt, (i[1], i[0]), False)
            # print(inside)
            if inside != -1:
                # print(i)
                cnt_points.append(i)
                edit_black.append((i[0], i[1]))
                ax[6].scatter(i[1], i[0])
                ax[7].scatter(i[1], i[0])
                ax[10].scatter(i[1], i[0])
                ax[11].scatter(i[1], i[0])
                # ax[8].scatter(i[1],i[0])
                cnts_x.append(i[1])
                cnts_y.append(i[0])

        all_cnt_p.append(cnt_points)

        # min/max points
        minX = min(cnts_x)
        maxX = max(cnts_x)
        minY = min(cnts_y)
        maxY = max(cnts_y)

        for i in ext:
            ax[6].scatter(i[0], i[1], color='r')

        ##        ax[6].scatter(minX,0, color='r')
        ##        ax[6].scatter(0,minY, color='r')
        ##        ax[6].scatter(maxX,0, color='r')
        ##        ax[6].scatter(0,maxY, color='r')

        x = cnts_x

        y = cnts_y
        points = np.c_[x, y]
        # print(len(points))
        from sklearn.neighbors import NearestNeighbors

        clf = NearestNeighbors(2).fit(points)
        G = clf.kneighbors_graph()

        import networkx as nx

        T = nx.from_scipy_sparse_matrix(G)
        order = list(nx.dfs_preorder_nodes(T, 0))
        # print(order)
        # xx = x[order]
        xx = [x[i] for i in order]
        # yy = y[order]
        yy = [y[i] for i in order]

        # ax[7].plot(xx, yy)
        ax[4].imshow(color_im)

        paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
        # print(len(paths))
        mindist = np.inf
        minidx = 0

        for i in range(len(points)):
            p = paths[i]  # order of nodes
            # print(p)
            ordered = points[p]  # ordered nodes
            # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
            if cost < mindist:
                mindist = cost
                minidx = i

        opt_order = paths[minidx]
        # print(len(opt_order))

        # xx = x[opt_order]
        xx = [x[i] for i in opt_order]
        # yy = y[opt_order]
        yy = [y[i] for i in opt_order]
        ax[8].plot(xx, yy)
        ax[8].scatter(xx[0], yy[0])
        ends.append((xx[0], yy[0]))
        ax[8].scatter(xx[-1], yy[-1])
        ends.append((xx[-1], yy[-1]))



all_points = [sum(v != 0 for v in i) for i in all_cnt_p]
ax[4].set_xlabel('Contour count = ' + str(len(all_cnt_p)))
ax[5].set_xlabel('Points count per cont = ' + str(all_points))

thin_bw = np.zeros_like(cv_skel)
for x in edit_black:
    thin_bw[x] = 255


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


goodcorners = cv2.goodFeaturesToTrack(thin_bw, 25, .05, 5)
cornerlist = []
goodcorners = np.int0(goodcorners)  # cast corners to int, needed to plot
ccount = 0
for i in goodcorners:
    print(str(i))
    x,y = i[0]
    cornerlist.append([x,y])
    ax[10].scatter(i[0][0], i[0][1], color='r')
    ax[9].set_xlabel('Corner count = ' + str(len(goodcorners)))
    ax[9].imshow(thin_bw)
    ax[9].scatter(i[0][0], i[0][1], color='r')
    ax[10].text((i[0][0]), (i[0][1] - 10), str(ccount), fontsize=12)
    ccount+=1
corner_connections = []
corner_combo = itertools.combinations(goodcorners, 2)
for i in corner_combo:
    points = [tup for tup in itertools.product(i[0], i[1])]
    corner_connections.append(points[np.argmin([distance(Pa, Pb) for (Pa, Pb) in points])])


# check_img = cv_skel.copy()
# check_img = check_img.astype(np.uint8)
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # kernel to dialate
# # check_img = cv2.dilate(check_img, kernel, iterations=1)  # dialate
# line_img = np.ones_like(color_im)*255
# cv2.imshow("Check_img", check_img)
boxes = []
gaps = []
for i in corner_connections:
    single_gap = []
    x, y = i
    dist = (distance(x, y))
    if dist < 30.0:
        min_x = min(x[0], y[0])  # left
        min_y = min(x[1], y[1])  # top
        max_x = max(x[0], y[0])
        max_y = max(x[1], y[1])
        width = max(x[0], y[0]) - min_x
        height = max(x[1], y[1]) - min_y
        rect = Rectangle((min_x, min_y), width,  height)
        boxes.append(rect)
        #if x[0] == y[0] or x[1] == y[1]:
        inside = [point for point in edit_black if (min_x <= point[0] <= min_x + width and min_y <= point[1] <= min_y + height)
                  or (min_x <= point[1] <= min_x + width and min_y <= point[0] <= min_y + height)]
        # else:
        #     inside = [point for point in edit_black if (min_x < point[0] < min_x + width and min_y < point[1] < min_y + height)
        #           or (min_x < point[1] < min_x + width and min_y < point[0] < min_y + height)]
        #print(inside)
        #print(inside)
        for k in inside:
            ax[10].scatter(k[1],k[0], color='g')
        # bin_line = np.zeros_like(check_img)  # create a matrix to draw the line in
        # print("CORNER", i)
        # points = line_points(x, y)
        # cv2.line(bin_line, tuple(x), tuple(y), color=255, thickness=1)  # draw line
        # conj = (check_img / 255 + bin_line / 255)  # create agreement image
        # n_agree = np.sum(conj == 2)
        # n_wrong = np.sum(conj == 1)
        # print("CONJ=", np.sum(conj))
        # agreement = (n_agree / (len(points)))
        # print(agreement)
        # if agreement <= 0.2:
        if len(inside) <= 3:
            single_gap.append((x[0], x[1]))
            single_gap.append((y[0], y[1]))

            ax[11].plot((x[0], y[0]), (x[1], y[1]), color='r')
            ax[11].scatter(x[0], x[1], color='r')
            ax[11].scatter(y[0], y[1], color='r')
        if single_gap != []:
            gaps.append(single_gap)
pc = PatchCollection(boxes, facecolor='r', alpha=0.25, edgecolor='b')
ax[10].add_collection(pc)
cv2.imshow("skel", cv_skel)
open_connections = []
combo = itertools.combinations(all_cnt_p, 2)
for i in combo:
    points = [tup for tup in itertools.product(i[0], i[1])]
    open_connections.append(points[np.argmin([distance(Pa, Pb) for (Pa, Pb) in points])])

for i in open_connections:
    x, y = i
    #print(distance(x, y))
    if dist < 20.0:
        ax[7].plot((x[1], y[1]), (x[0], y[0]), color='r')
        ax[7].scatter(x[1], x[0], color='r')
        ax[7].scatter(y[1], y[0], color='r')

corner_combo = itertools.combinations(goodcorners, 2)
# # for i in combo
# with open(os.path.basename(path) + "_SampleOutput" + ".txt", "w") as text_file:
#     text_file.write("MMSE Pentagon GUI Version=1\n")
#     #text_file.write("Process date = {0}\n".format(datetime.datetime.now()))
#     text_file.write("corner count =  {0}\n".format(str(len(cornerlist))))
#     text_file.write("line count =  {0}\n".format(str(len(lines))))
#     text_file.write("corner list =  {0}\n".format(str(cornerlist)))
#     text_file.write("lines list =  {0}\n".format(str(lines)))
#     text_file.write("gaps list =  {0}\n".format(str(gaps)))
#
# with open(os.path.basename(path) + "_cornerlist_output.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(cornerlist)
#
# with open(os.path.basename(path) + "_linelist_output.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     for i in lines:
#         writer.writerows([i[0],i[1]])
#
# with open(os.path.basename(path) + "_gapslist_output.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     for i in gaps:
#         writer.writerows([i[0],i[1]])


ax[5].invert_yaxis()
ax[6].invert_yaxis()
ax[7].invert_yaxis()
ax[8].invert_yaxis()
#ax[9].invert_yaxis()
ax[10].invert_yaxis()
ax[11].invert_yaxis()
plt.tight_layout()
plt.show()


