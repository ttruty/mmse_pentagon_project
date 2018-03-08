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
from skimage.measure import compare_ssim
import imutils
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist



warnings.filterwarnings("ignore", category=DeprecationWarning)

path = tkinter.filedialog.askopenfilename()
tkinter.wantobjects
color_im = cv2.imread(path)
img = cv2.cvtColor(color_im, cv2.COLOR_RGB2GRAY)

color_im = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
clone = color_im.copy()
##thresh = threshold_otsu(img)
##binary = img > thresh
inv_img = np.invert(img)
binary = binarize(inv_img, threshold=8.0)
#binary = np.invert(binary)
skel = skeletonize(binary)

##plt.imshow(skel)
##plt.show()
cv_skel = img_as_ubyte(skel)



lines = probabilistic_hough_line(skel,
                                 threshold=1,
                                 line_length=2,
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
             cmap=cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(skel, cmap=cm.gray)
line_count = 0
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - skel.shape[1] * np.cos(angle)) / np.sin(angle)
    #print(angle, dist)
    line_count += 1
    ax[2].plot((0, skel.shape[1]), (y0, y1), '-r')
ax[2].set_xlim((0, skel.shape[1]))
ax[2].set_ylim((skel.shape[0], 0))
ax[2].set_xlabel('Line count = ' + str(line_count))
#ax[2].set_axis_off()
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

im2, contours, hierarchy = cv2.findContours(cv_skel,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #
##cv2.drawContours(color_im, contours, -1, (0,0,255), 3)
##plt.imshow(color_im)
##plt.show()
all_cnt_p = []
x = match_p[0]
y = match_p[1]
black_coords = list(zip(x,y))
ax[5].scatter(y,x)
edit_black = []
#print(black_coords)
cnt_count = 0
ends = []
ext_points = []
for cnt in contours:
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(hull)
    #print(area)
    perimeter = cv2.arcLength(hull,True)
    if area > 25:
        #print("MAX  ", cnt)
        # determine the most extreme points along the contour
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
        ext = [extLeft, extRight, extTop, extBot]
        ext_points.append(ext)
        
        #print("L: ", extLeft, "R: ", extRight, "T: ", extTop, "B: ", extBot)
        cv2.circle(color_im, extLeft, 6, (0, 0, 255), -1)
        cv2.circle(color_im, extRight, 6, (0, 255, 0), -1)
        cv2.circle(color_im, extTop, 6, (255, 0, 0), -1)
        cv2.circle(color_im, extBot, 6, (255, 255, 0), -1)
        
        cnts_x = []
        cnts_y = []
        #print("CNT Count = ", cnt_count)
        color = np.random.uniform(0,255,3)
        cv2.drawContours(color_im, cnt, -1, color, 3)
        cnt_points = []
        #print(len(black_coords))
        for i in black_coords:
            #print(i)
            inside = cv2.pointPolygonTest(cnt, (i[1],i[0]),False)
            #print(inside)
            if inside != -1:
                #print(i)
                cnt_points.append(i)
                edit_black.append((i[0],i[1]))
                ax[6].scatter(i[1],i[0])
                ax[7].scatter(i[1],i[0])
                ax[10].scatter(i[1],i[0])
                ax[11].scatter(i[1],i[0])
                #ax[8].scatter(i[1],i[0])
                cnts_x.append(i[1])
                cnts_y.append(i[0])


        all_cnt_p.append(cnt_points)
        
        #min/max points
        minX = min(cnts_x)
        maxX = max(cnts_x)
        minY = min(cnts_y)
        maxY = max(cnts_y)

        for i in ext:
            ax[6].scatter(i[0],i[1], color='r')
        
##        ax[6].scatter(minX,0, color='r')
##        ax[6].scatter(0,minY, color='r')
##        ax[6].scatter(maxX,0, color='r')
##        ax[6].scatter(0,maxY, color='r')
        
        x = cnts_x
        
        y = cnts_y
        points = np.c_[x, y]
        #print(len(points))
        from sklearn.neighbors import NearestNeighbors

        clf = NearestNeighbors(2).fit(points)
        G = clf.kneighbors_graph()

        import networkx as nx

        T = nx.from_scipy_sparse_matrix(G)
        order = list(nx.dfs_preorder_nodes(T, 0))
        #print(order)
        #xx = x[order]
        xx = [x[i] for i in order]
        #yy = y[order]
        yy = [y[i] for i in order]
        
        #ax[7].plot(xx, yy)
        ax[4].imshow(color_im)

        paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
        #print(len(paths))
        mindist = np.inf
        minidx = 0

        for i in range(len(points)):
            p = paths[i]           # order of nodes
            #print(p)
            ordered = points[p]    # ordered nodes
            # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
            if cost < mindist:
                mindist = cost
                minidx = i
         
        opt_order = paths[minidx]
        #print(len(opt_order))

        #xx = x[opt_order]
        xx = [x[i] for i in opt_order]
        #yy = y[opt_order]
        yy = [y[i] for i in opt_order]
        ax[8].plot(xx, yy)
        ax[8].scatter(xx[0],yy[0])
        ends.append((xx[0],yy[0]))
        ax[8].scatter(xx[-1],yy[-1])
        ends.append((xx[-1],yy[-1]))

        ax[5].invert_yaxis()
        ax[6].invert_yaxis()
        ax[7].invert_yaxis()
        ax[8].invert_yaxis()
        ax[9].invert_yaxis()
        ax[10].invert_yaxis()
        ax[11].invert_yaxis()
        
all_points = [sum(v != 0 for v in i) for i in all_cnt_p]
ax[4].set_xlabel('Contour count = ' + str(len(all_cnt_p)))
ax[5].set_xlabel('Points count per cont = ' + str(all_points))

thin_bw = np.zeros_like(cv_skel)
for x in edit_black:
    thin_bw[x] = 255

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

goodcorners = cv2.goodFeaturesToTrack(thin_bw,50,.20,2)
goodcorners = np.int0(goodcorners) # cast corners to int, needed to plot

for i in goodcorners:
    print(i)
    ax[10].scatter(i[0][0],i[0][1], color = 'r')
    ax[9].set_xlabel('Corner count = ' + str(len(goodcorners)))
    ax[9].imshow(thin_bw)
    ax[9].scatter(i[0][0], i[0][1], color='r')
corner_connections = []
corner_combo = itertools.combinations(goodcorners,2)
for i in corner_combo:
    points = [tup for tup in itertools.product(i[0], i[1])]
    corner_connections.append(points[np.argmin([distance(Pa, Pb) for (Pa, Pb) in points])])

for i in corner_connections:
    x,y = i
    print(i)
    print(distance(x,y))
    dist = (distance(x,y))
    if dist < 20.0:
        ax[11].plot((x[0],y[0]),(x[1],y[1]),color='r')
        ax[11].scatter(x[0],x[1],color='r')
        ax[11].scatter(y[0],y[1],color='r')

open_connections = []  
combo = itertools.combinations(all_cnt_p,2)
for i in combo:
    points = [tup for tup in itertools.product(i[0], i[1])]
    open_connections.append(points[np.argmin([distance(Pa, Pb) for (Pa, Pb) in points])])

for i in open_connections:
    x,y = i
    print(i)
    print(distance(x,y))
    dist = (distance(x,y))
    if dist < 20.0:
        ax[7].plot((x[1],y[1]),(x[0],y[0]),color='r')
        ax[7].scatter(x[1],x[0],color='r')
        ax[7].scatter(y[1],y[0],color='r')

corner_combo = itertools.combinations(goodcorners,2)
#for i in combo


plt.tight_layout()
plt.show()
    



