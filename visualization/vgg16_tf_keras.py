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

import keras
import tensorflow as tf
from keras import backend as K
from keras import applications
from keras.models import Model

#from sklearn.manifold import TSNE
#from sklearn import manifold
from sklearn import cluster

# prototype.

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
    kmeans = cluster.KMeans(n_clusters=cluster_num, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                           precompute_distances='auto', verbose=0, random_state=None, copy_x=True,
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

if __name__ == "__main__":
    ## load the pretrained model
    model = applications.VGG16(weights='imagenet', include_top=True)

    # input image and get hypercolumns
    images_dir = 'generated_1/'
    input_images = read_and_preprocess_dir(images_dir)
    IMG_NUM = len(input_images)
    positives_num = 90
    negatives_num = 10
    # layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    layers = ['block4_conv3']

    if 0:
        #################################### sift+hypercolumns ###################################
        Kpts_per_img = []
        desc_list = []

        for i in range(IMG_NUM):
            img = input_images[i][0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
            kp, _ = sift.detectAndCompute(gray, None)
            Kpts_per_img.append(len(kp))

            hc = extract_hypercolumn(model, layers, input_images[i], False)
            hc = hc.transpose(1, 2, 0)

            for j in range(len(kp)):
                x = int(math.floor(kp[j].pt[0]))
                y = int(math.floor(kp[j].pt[1]))
                des = hc[x][y]  # here not sure it is x,y or y,x
                desc_list.append(des)

        print len(desc_list)
        desc_array = np.asarray(desc_list)
        desc_all = desc_array.astype(np.float64)

    if 1:
        #################################### sift ###################################
        Kpts_per_img = []
        desc_list = []

        for i in range(IMG_NUM):
            img = input_images[i][0]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
            kp, des = sift.detectAndCompute(gray, None)
            #img = cv2.drawKeypoints(gray, kp)
            #cv2.imwrite('sift_featured/sift_keypoints_'+str(i)+'.jpg', img)
            Kpts_per_img.append(len(kp))
            desc_list.append(des)

        desc_array = np.array(desc_list)
        desc_temp = desc_array[0]
        for i in range(desc_array.shape[0]):
            if (i+1) < desc_array.shape[0]:
                desc_temp = np.concatenate((desc_temp, desc_array[i+1]), axis=0)

        desc_all = desc_temp.astype(np.float64)

        #######################################################################################################

    if 0:
        ######################################################################################################
        # kmeans-->global (fails becauese of memory)
        hc_array_1 = hc_array.reshape(hc_array.shape[0], hc_array.shape[1], hc_array.shape[2]*hc_array.shape[3])
        Kpts_per_img = hc_array_1.shape[2]
        hc_array_2 = hc_array_1.transpose(0, 2, 1)

        hc_temp = hc_array_2[0]
        for j in range(hc_array_2.shape[0]):
            if (j+1) < hc_array_2.shape[0]:
                hc_temp = np.concatenate((hc_temp,hc_array_2[j+1]), axis=0)
        hc_ready = hc_temp.astype(np.float64)
        cluster_num = 4
        cluster_labels, cluster_centers = kmeans_hypercolumns(hc_ready, cluster_num)

    if 0:
        ########################################extract random keypoints###################################################
        # random_size keypoints per image
        random_size = 300
        randx_ = [randint(0, 223) for i in range(random_size)]
        randy_ = [randint(0, 223) for i in range(random_size)]

        hc_array_temp = [hc_array[:,:,randx_[i],randy_[i]] for i in range(random_size)]
        hc_array_1 = np.asarray(hc_array_temp)
        hc_array_2 = hc_array_1.transpose(1, 0, 2)

        #print hc_array_2.shape

        hc_temp = hc_array_2[0]
        for j in range(hc_array_2.shape[0]):
            if (j + 1) < IMG_NUM:
                hc_temp = np.concatenate((hc_temp, hc_array_2[j + 1]), axis=0)

        #print hc_temp.shape

        hc_ready = hc_temp.astype(np.float64)
        cluster_num = 256
        cluster_labels, cluster_centers = kmeans_hypercolumns(hc_ready, cluster_num)
        Kpts_per_img = np.zeros((IMG_NUM,))
        for i in range(IMG_NUM):
            Kpts_per_img[i] = random_size
        ########################################################################################################
    if 0:
        ####################################extract keypoints for each image###################################
        hc_array_1 = hc_array.reshape(hc_array.shape[0], hc_array.shape[1], hc_array.shape[2]*hc_array.shape[3])
        hc_array_2 = hc_array_1.transpose(0, 2, 1)
        Kpts_per_img = np.zeros((IMG_NUM,))
        extracted_num = 200
        for i in range(IMG_NUM):
            Kpts_per_img[i] = extracted_num

        keypts_extracted = []
        for i in range(hc_array_2.shape[0]):
            hc_ready = hc_array_2[i].astype(np.float64)
            cluster_num = extracted_num
            cluster_labels, cluster_centers = kmeans_hypercolumns(hc_ready, cluster_num)
            keypts_extracted.append(cluster_centers)

        hc_array_2_temp = np.asarray(keypts_extracted)
        hc_temp = hc_array_2_temp[0]
        for j in range(hc_array_2_temp.shape[0]):
            if (j + 1) < IMG_NUM:
                hc_temp = np.concatenate((hc_temp, hc_array_2_temp[j + 1]), axis=0)
        hc_ready_temp = hc_temp.astype(np.float64)
        cluster_num = 256
        cluster_labels, cluster_centers = kmeans_hypercolumns(hc_ready_temp, cluster_num)
        #######################################################################################################
    if 0:
        #################################### sift+data from matlab 4 debug###################################
        # load descriptors from matlab
        mat = scipy.io.loadmat('data.mat')
        data = np.transpose(mat['data'])

        mat_1 = scipy.io.loadmat('des_num.mat')
        des_num = mat_1['des_num'][0]

        Kpts_per_img = []
        for i in range(des_num.shape[0]):
            Kpts_per_img.append(int(des_num[i]))

        desc_all = data.astype(np.float64)

        ##############################################################################################
    cluster_num = [2048, 4096, 8192]
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

                kk = positives_num-1
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

        # plot ROC curve
        TPR = np.asarray(TPR)
        FPR = np.asarray(FPR)
        fig = plt.figure()
        plt.plot(FPR[0,:], TPR[0,:], linestyle='-', color='r',label='5%')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[1,:], TPR[1,:], linestyle='-', color='g', label='10%')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[2,:], TPR[2,:], linestyle='-', color='b', label='25%')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[3,:], TPR[3,:], linestyle='-', color='c', label='50%')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[4,:], TPR[4,:], linestyle='-', color='m', label='60%')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[5,:], TPR[5,:], linestyle='-', color='y', label='70%')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[6,:], TPR[6,:], linestyle='-', color='k', label='80%')
        plt.axis([0, 1, 0, 1])
        plt.plot(FPR[7,:], TPR[7,:], linestyle='-.', color='r', label='90%')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig('/Users/nanliu/Desktop/v90_blcok4_conv3_'+str(cluster_num[cc])+'.png')
        plt.show()
        plt.close(fig)


# test with single image
    #model = applications.VGG16(weights='imagenet', include_top=True)
    #file_path = '/Users/nanliu/PycharmProjects/vgg/data/madruga.jpeg'
    #im = read_and_preprocess(file_path)
    ##layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    #layers = ['block1_conv2']
    #hc = extract_hypercolumn(model, layers, im)
    #plot_avarage_hypercolumns(hc)
    #hc = hc.transpose(1, 2, 0).reshape(50176, -1)
    #_,_= kmeans_hypercolumns(hc, 2)
