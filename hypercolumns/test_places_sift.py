import cv2
import os
import re
import math
import sys
import numpy as np
import scipy as sp
import scipy.io

from random import randint
from operator import itemgetter
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

import keras
import tensorflow as tf
from keras import backend as K
from keras import applications
from keras.models import Model
from sklearn import cluster

# SIFT based method for MITPlaces datasets.

root_dir = '/home/users/lui/Desktop/hypercolumns_MITPlaces/MITPlaces/'
img_dir = os.path.join(root_dir, 'data/')

# read images
def read_and_preprocess(file_path):
    img = cv2.imread(file_path)
    im = np.expand_dims(img, axis=0)

    return im

## read images from name_list
def read_and_preprocess_txt(name_list):
    """
    Load image from the filename. Default is to load in color if
    possible.

    Args:
        img_name (string): string of the image name, relative to
            the image directory.

    Returns:
        cv::Mat
    """
    names = open(name_list, 'r')
    lines = names.read().splitlines()
    inp_imgs = []
    for item in lines:
        img_name = os.path.join(img_dir, item + '.jpg')
        inp_img = read_and_preprocess(img_name)
        inp_imgs.append(inp_img)
    return inp_imgs

## k-means clustering
## m=shape(n_samples, n_features)
def kmeans_hypercolumns(hc, cluster_num):

    kmeans = cluster.MiniBatchKMeans(n_clusters=cluster_num,init='k-means++', max_iter = 100, batch_size = 3000,
                                     verbose = 1, n_init = 3).fit(hc)
    return kmeans.labels_, kmeans.cluster_centers_

def indices(a, func):
    return [int(i) for (i, val) in enumerate(a) if func(val)]

def ismember(A, B):
    return [ np.sum(a == B) for a in A ]

if __name__ == "__main__":

    file_folder = '/home/users/lui/Desktop/hypercolumns_MITPlaces/MITPlaces/attic/'
    images = read_and_preprocess_txt(os.path.join(file_folder, 'test.txt'))
    IMG_NUM = len(images)
    positives_num = 500
    negatives_num = 50

    #################################### sift ###################################
    Kpts_per_img = []
    desc_list = []

    for i in range(IMG_NUM):
        img = images[i][0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        kp, des = sift.detectAndCompute(gray, None)
        # img = cv2.drawKeypoints(gray, kp)
        # cv2.imwrite('visualization/sift_keypoints_'+str(i)+'.jpg', img)
        if int(len(kp)) == 0:
            if i < positives_num:
                positives_num = positives_num - 1
            else:
                negatives_num = negatives_num -1
        else:
            Kpts_per_img.append(len(kp))
            desc_list.append(des)
    IMG_NUM = positives_num + negatives_num
    # print('positives_num = %d' % positives_num)
    # print('negatives_num = %d' % negatives_num)
    desc_array = np.array(desc_list)
    desc_temp = desc_array[0]

    for i in range(desc_array.shape[0]):
        if (i+1) < desc_array.shape[0]:
            desc_temp = np.concatenate((desc_temp, desc_array[i+1]), axis=0)

    desc_all = desc_temp.astype(np.float64)
    print desc_all.shape
    ################################################################################
    # [16384, 32768, 65536, 131072, 262144]
    cluster_num = [8192]
    for cc in range(len(cluster_num)):
        cluster_labels, cluster_centers = kmeans_hypercolumns(desc_all, cluster_num[cc])

        keypts_total = cluster_labels.shape[0]
        image_labels = np.zeros((keypts_total,))
        size_sum = 0
        size_cnt = 0
        for i in range(IMG_NUM):
            size_cnt = int(Kpts_per_img[i])
            if i == 0:
                image_labels[0:size_cnt] = i
            else:
                image_labels[size_sum: size_sum+size_cnt] = i
            size_sum = size_sum + size_cnt

        # clusters
        clusters = []
        for center in range(cluster_num[cc]):
            temp = {}
            temp['center'] = cluster_centers[center]
            inds = indices(cluster_labels, lambda x: x == center)
            temp['keyptInd'] = [inds]
            temp['imageInd'] = [int(image_labels[i]) for i in inds]
            temp['keyptNum'] = len(inds)
            clusters.append(temp)

        sorted_clusters = sorted(clusters, key = itemgetter('keyptNum'), reverse=True)

        #for i in range(cluster_num[cc]):
        #    print sorted_clusters[i]['keyptNum']

        # thresholding
        nclusters = [math.ceil(cluster_num[cc]*0.05), math.ceil(cluster_num[cc]*0.1), math.ceil(cluster_num[cc]*0.25),
                     math.ceil(cluster_num[cc]*0.5), math.ceil(cluster_num[cc]*0.6), math.ceil(cluster_num[cc]*0.7),
                     math.ceil(cluster_num[cc]*0.8), math.ceil(cluster_num[cc]*0.9)]

        TPR = []
        FPR = []
        for n_c in range(len(nclusters)):
            nc = int(nclusters[n_c])
            TP = []
            FP = []
            for tao in np.arange(0.0, 1.1, 0.02):
                #print('tao = ', tao)
                flag = np.zeros((IMG_NUM,))
                for i in range(IMG_NUM):
                    #count = 0
                    #for j in range(nc):
                    #    img_ind = indices(sorted_clusters[j]['imageInd'], lambda x: x == i)
                    #    count = count + len(img_ind)
                    #Inlier_per_img = count

                    sel_pos = []
                    for j in range(nc):
                        img_ind = indices(sorted_clusters[j]['imageInd'], lambda x: x == i)
                        for cnt in range(len(img_ind)):
                            sel_pos.append(sorted_clusters[j]['keyptInd'][0][img_ind[cnt]])
                    Inlier_per_img = len(sel_pos)
                    if (Inlier_per_img*1.0/Kpts_per_img[i]) >= tao:
                        flag[i] = 1

                #print flag

                pos_ind = len(indices(flag, lambda x: x == 1))
                #print('number of detected as true: ', pos_ind)
                k = 0
                tp_count = 0
                while(k < positives_num):
                    if flag[k] == 1:
                        tp_count = tp_count + 1
                    k = k+1
                tp = tp_count

                kk = positives_num
                fp_count = 0
                while (kk < positives_num+negatives_num):
                    if flag[kk] == 1:
                        fp_count = fp_count + 1
                    kk = kk + 1
                fp = fp_count

                TP.append(tp*1.0/positives_num)
                FP.append(fp*1.0/negatives_num)
            TPR.append(TP)
            FPR.append(FP)


        file = open('/home/users/lui/Desktop/hypercolumns_MITPlaces/txt_results/attic_sift_8192.txt', 'w')
        for j in range(8):
            file.write("%s" % FPR[j][0])
            for i in range(50-1):
                file.write(",")
                file.write("%s" % FPR[j][i + 1])
            file.write("\n")
            file.write("%s" % TPR[j][0])
            for i in range(50-1):
                file.write(",")
                file.write("%s" % TPR[j][i + 1])
            file.write("\n")

        # plot ROC curve
        TPR = np.asarray(TPR)
        FPR = np.asarray(FPR)
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.plot(FPR[0,:], TPR[0,:], linestyle='-', color='r',label='5% clusters')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[1,:], TPR[1,:], linestyle='-', color='g', label='10% clusters')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[2,:], TPR[2,:], linestyle='-', color='b', label='25% clusters')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[3,:], TPR[3,:], linestyle='-', color='c', label='50% clusters')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[4,:], TPR[4,:], linestyle='-', color='m', label='60% clusters')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[5,:], TPR[5,:], linestyle='-', color='y', label='70% clusters')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[6,:], TPR[6,:], linestyle='-', color='k', label='80% clusters')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[7,:], TPR[7,:], linestyle='-', color='0.75', label='90% clusters')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('SIFT ROC')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])
        # ax.set_axisbelow(True)
        # ax.xaxis.grid(color='gray', linestyle='solid')
        # ax.yaxis.grid(color='gray', linestyle='solid')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False, prop={'size': 7})
        fig.savefig(file_folder+'attic_sift_'+str(cluster_num[cc]), bbox_inches='tight')

