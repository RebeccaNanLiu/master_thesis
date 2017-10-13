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
from sklearn import cluster

import keras
import tensorflow as tf
from keras import backend as K
from keras import applications
from keras.models import Model

# visualization of codewords for Hypercolumn and SIFT based methods.

# read images
def read_and_preprocess(file_path):
    img = cv2.imread(file_path)

    # commented out for the test of sift
    im_original = cv2.resize(img, (224, 224))
    im = np.expand_dims(im_original, axis=0)

    #im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
    #plt.figure()
    #plt.imshow(im_converted)
    #plt.axis('off')
    #plt.show()

    return im

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
            if cnt==1:
                convmap = np.expand_dims(convmap, axis = 0)
                feat_map = convmap.transpose((2, 0, 1)) # convmap for fc2
            else:
                feat_map = convmap[0].transpose((2, 0, 1)) # convmap[0] for convlayers
            for fmap in feat_map:
                upscaled = sp.misc.imresize(fmap, size=(224,224), mode='F', interp='bilinear')
                hypercolumns.append(upscaled)
    else:
        for convmap in feature_maps:
            if flag:
                convmap = np.expand_dims(convmap, axis = 0)
                convmap = np.expand_dims(convmap, axis = 0)
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
    kmeans = cluster.KMeans(n_clusters=cluster_num, init='k-means++', n_init=1, max_iter=100, tol=0.0001,
                           precompute_distances='auto', verbose=1, random_state=None, copy_x=True,
                            n_jobs=1, algorithm='auto').fit(hc)
    #cluster_labels = kmeans.fit_predict(hc)
    #imcluster = np.zeros((224, 224))
    #imcluster = imcluster.reshape((224 * 224,))
    #imcluster = cluster_labels
    #plt.figure()
    #plt.imshow(imcluster.reshape(224, 224), cmap="hot")
    #plt.axis('off')
    #plt.show()

    return kmeans.labels_, kmeans.cluster_centers_

def indices(a, func):
    return [int(i) for (i, val) in enumerate(a) if func(val)]

def ismember(A, B):
    return [ np.sum(a == B) for a in A ]

def getfeatures(pts_array, ind_list):
    features = []
    for i in range(len(ind_list)):
            features.append(pts_array[ind_list[i]])
    return features

if __name__ == "__main__":
    model = applications.VGG16(weights='imagenet', include_top=True)

    images_dir = '/Users/nanliu/hypercolumns/generated/generated_90/'
    input_images = read_and_preprocess_dir(images_dir)
    IMG_NUM = len(input_images)
    positives_num = 90
    negatives_num = 10
    layers = ['block5_conv3']

    if 0:
        #################################### sift ###################################
        Kpts_per_img = []
        kps_list = []
        desc_list = []

        for i in range(IMG_NUM):
            img = input_images[i][0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
            kp, des = sift.detectAndCompute(gray, None)
            Kpts_per_img.append(len(kp))
            desc_list.append(des)
            kps_list.append(kp)

        desc_array = np.asarray(desc_list)
        kps_array = np.asarray(kps_list)

        kps_temp = kps_array[0]
        for i in range(kps_array.shape[0]):
            if (i+1) < desc_array.shape[0]:
                kps_temp = np.concatenate((kps_temp, kps_array[i+1]), axis=0)
        kps_all = kps_temp

        desc_temp = desc_array[0]
        for i in range(desc_array.shape[0]):
            if (i+1) < desc_array.shape[0]:
                desc_temp = np.concatenate((desc_temp, desc_array[i+1]), axis=0)

        desc_all = desc_temp.astype(np.float64)
        print desc_all.shape
        ################################## hypercolumn ########################################
    if 1:
        Kpts_per_img = []
        kps_list = []
        desc_list = []

        for i in range(IMG_NUM):
            img = input_images[i][0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
            kp, _ = sift.detectAndCompute(gray, None)
            Kpts_per_img.append(len(kp))
            kps_list.append(kp)

            hc = extract_hypercolumn(model, layers, input_images[i], False)
            hc = hc.transpose(1, 2, 0)

            for j in range(len(kp)):
                x = int(math.floor(kp[j].pt[0]))
                y = int(math.floor(kp[j].pt[1]))
                des = hc[x][y]  # here not sure it is x,y or y,x
                desc_list.append(des)

        desc_array = np.asarray(desc_list)
        kps_array = np.asarray(kps_list)

        kps_temp = kps_array[0]
        for i in range(kps_array.shape[0]):
            if (i + 1) < kps_array.shape[0]:
                kps_temp = np.concatenate((kps_temp, kps_array[i + 1]), axis=0)
        kps_all = kps_temp

        desc_all = desc_array.astype(np.float64)
        print desc_all.shape
    #########################################################################################

    cluster_num = [2048]
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

        text_file = open('/Users/nanliu/hypercolumn-visualization/hypercolumn_hist.txt', 'w')
        for i in range(2048):
            t = sorted_clusters[i]['keyptNum']
            text_file.write("%s\n" % t)
        sys.exit()


        nclusters = [0, 1, 2, 3, 4]


        for i in range(IMG_NUM):
            img = input_images[i][0]
            for n_c in range(len(nclusters)):
                sel_pos = []
                img_ind = indices(sorted_clusters[nclusters[n_c]]['imageInd'], lambda x: x == i)
                for cnt in range(len(img_ind)):
                    sel_pos.append(sorted_clusters[nclusters[n_c]]['keyptInd'][0][img_ind[cnt]])

                kpts = getfeatures(kps_all, sel_pos)
                if n_c == 0:
                    img = cv2.drawKeypoints(img, kpts, color=(0,0,255)) #blue
                    # plt.imshow(img), plt.show()
                if n_c == 1:
                    img = cv2.drawKeypoints(img, kpts, color=(0,255,0)) #green
                    # plt.imshow(img), plt.show()
                if n_c == 2:
                    img = cv2.drawKeypoints(img, kpts, color=(0,0,255))
                    # plt.imshow(img), plt.show()
                if n_c == 3:
                    img = cv2.drawKeypoints(img, kpts, color=(0,255,255))
                    # plt.imshow(img), plt.show()
                if n_c == 4:
                    img = cv2.drawKeypoints(img, kpts, color=(255,255,0))
                    # plt.imshow(img), plt.show()
            cv2.imwrite('sift_featured/featured_90/'+str(i)+'.jpg', img)

