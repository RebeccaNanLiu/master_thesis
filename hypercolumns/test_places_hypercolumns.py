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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage import io

import keras
import tensorflow as tf
from keras import backend as K
from keras import applications
from keras.applications import imagenet_utils
from keras.models import Model
from sklearn import cluster

# Hypercolumn based method for MITPlaces datasets.

root_dir = '/home/users/lui/Desktop/hypercolumns_MITPlaces/MITPlaces/'
img_dir = os.path.join(root_dir, 'data/')

inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# read images
def read_and_preprocess(file_path):
    img = cv2.imread(file_path)
    im_original = cv2.resize(img, inputShape)
    im = np.expand_dims(im_original, axis=0)
    #im = preprocess(im)
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

## read images from directory
def read_and_preprocess_dir(images_dir):
    list_images = [images_dir + f for f in os.listdir(images_dir) if re.search('jpeg|JPEG', f)]
    list_images = sorted(list_images, key=lambda x: (int(re.sub('\D', '', x)), x))

    inp_imgs = []
    inp_size = len(list_images)
    for id in range(inp_size):
        inp_im = read_and_preprocess(list_images[id])
        inp_imgs.append(inp_im)
    return inp_imgs

## prediction of im in vgg-16
def predict_histogram(model, instance):
    out = model.predict(instance)
    plt.figure()
    plt.plot(out.ravel())


## extract single feature maps
def extract_layer_output(model, layer_name, instance):
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(instance)
    feat = intermediate_output[0].transpose((2,0,1))

    plt.figure()
    plt.imshow(feat[0].astype(np.uint8))
    plt.axis('off')
    plt.show()

## extract hypercolumns
def extract_hypercolumn(model, layer_names, instance, flag):
    layers = [model.get_layer(ln).output for ln in layer_names]
    get_feature = Model(inputs=model.input, outputs=layers)
    feature_maps = get_feature.predict(instance)
    hypercolumns = []

    cnt = 0
    if len(feature_maps)>1 :
        for convmap in feature_maps:
            cnt = cnt + 1
            if cnt == 1 and flag == True:
                convmap = np.expand_dims(convmap, axis = 0)
                feat_map = convmap.transpose((2, 0, 1)) # convmap for fc2
            else:
                feat_map = convmap[0].transpose((2, 0, 1)) # convmap[0] for convlayers
            for fmap in feat_map:
                upscaled = sp.misc.imresize(fmap, size=(224,224), mode='F', interp='bilinear')
                hypercolumns.append(upscaled)
    else:
        for convmap in feature_maps:
            feat_map = convmap.transpose((2, 0, 1))
            for fmap in feat_map:
                upscaled = sp.misc.imresize(fmap, size=(224,224), mode='F', interp='bilinear')
                hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)

## plot the average of the hypercolumns activations for each pixel
def plot_avarage_hypercolumns(hc):
    ave = np.average(hc.transpose(1, 2, 0), axis=2)
    plt.figure()
    plt.imshow(ave)
    plt.axis('off')
    plt.show()

## k-means clustering
## m=shape(n_samples, n_features)
def kmeans_hypercolumns(hc, cluster_num):

    #print("shape of hc:", hc.shape)
    kmeans = cluster.KMeans(n_clusters=cluster_num, init='k-means++', n_init=3, max_iter=100, verbose=1).fit(hc)
    return kmeans.labels_, kmeans.cluster_centers_

def minibatch_kmeans_hypercolumns(hc, cluster_num):
    kmeans = cluster.MiniBatchKMeans(n_clusters=cluster_num,init='k-means++', max_iter = 100, batch_size = 3000,
                                     verbose = 1, n_init = 3).fit(hc)
    return kmeans.labels_, kmeans.cluster_centers_

def indices(a, func):
    return [int(i) for (i, val) in enumerate(a) if func(val)]

def ismember(A, B):
    return [ np.sum(a == B) for a in A ]

if __name__ == "__main__":

    ## load the pretrained model
    model = applications.VGG16(weights='imagenet', include_top=True)

    file_folder = '/home/users/lui/Desktop/hypercolumns_MITPlaces/MITPlaces/attic/'
    input_images = read_and_preprocess_txt(os.path.join(file_folder, 'test.txt'))

    IMG_NUM = len(input_images)
    positives_num = 500
    negatives_num = 50
    # layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    layers = ['block5_conv3']

    #################################### sift+hypercolumns ###################################
    Kpts_per_img = []
    desc_list = []

    for i in range(IMG_NUM):
        img = input_images[i][0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.06, edgeThreshold=10, sigma=1.6)
        kp, des = sift.detectAndCompute(gray, None)
        Kpts_per_img.append(len(kp))

        hc = extract_hypercolumn(model, layers, input_images[i], False)
        hc = hc.transpose(1,2,0)

        for j in range(len(kp)):
            x = int(math.floor(kp[j].pt[0]))
            y = int(math.floor(kp[j].pt[1]))
            des = hc[x][y] # here not sure it is x,y or y,x
            desc_list.append(des)

    desc_array = np.asarray(desc_list)
    desc_all = desc_array.astype(np.float64)
    print desc_all.shape

    # [16384, 32768, 65536, 131072, 262144]
    cluster_num = [8192]
    for cc in range(len(cluster_num)):
        cluster_labels, cluster_centers = minibatch_kmeans_hypercolumns(desc_all, cluster_num[cc])

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
            for tao in np.arange(0.0, 1.0, 0.02):
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

        file = open('/home/users/lui/Desktop/hypercolumns_MITPlaces/txt_results/attic_conv13_8192.txt', 'w')
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
        plt.title('Hypercolumns ROC')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])
        # ax.set_axisbelow(True)
        # ax.xaxis.grid(color='gray', linestyle='solid')
        # ax.yaxis.grid(color='gray', linestyle='solid')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=4,  frameon=False, prop={'size': 7})
        fig.savefig(file_folder + 'attic_conv13_' + str(cluster_num[cc]), bbox_inches='tight')
