import cv2
import os
import re
import math
import sys
import csv
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt
from keras import applications
from keras.models import Model
from sklearn import cluster

# Image Clustering method.

## load the pretrained model
model = applications.ResNet50(weights='imagenet', include_top=True, classes=1000)

root_dir = '/Users/nanliu/data/VOC2012/'
img_dir = os.path.join(root_dir, 'JPEGImages/')

# read images
def read_and_preprocess(file_path):
    img = cv2.imread(file_path)

    # commented out for the test of sift
    im_original = cv2.resize(img, (224, 224))
    im = np.expand_dims(im_original, axis=0)
    # im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
    # plt.figure()
    # plt.imshow(im_converted)
    # plt.axis('off')
    # plt.show()
    return im

## read images from name_list
def read_and_preprocess_txt(name_list):
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

## extract hypercolumns
def extract_hypercolumn(model, layer_names, instance):
    layers = [model.get_layer(ln).output for ln in layer_names]
    get_feature = Model(inputs=model.input, outputs=layers)
    feature_maps = get_feature.predict(instance)
    return feature_maps

## m=shape(n_samples, n_features)
def kmeans_hypercolumns(hc, cluster_num):

    kmeans = cluster.KMeans(n_clusters=cluster_num, init='k-means++', n_init=10, max_iter=100, verbose=1).fit(hc)
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

    save_folder = '/Users/nanliu/Desktop/image_clustering/'
    # input image and get hypercolumns
    file_folder = '/Users/nanliu/hypercolumns/pascal/car/'
    input_images = read_and_preprocess_txt(os.path.join(file_folder, 'test_20.txt'))
    # save_file = '/Users/nanliu/Desktop/image_clustering/aeroplane_2_128.txt'
    IMG_NUM = len(input_images)
    positives_num = 1161
    negatives_num = 232

    layers = ['avg_pool']
    desc_list = []

    for i in range(IMG_NUM):
        hc = extract_hypercolumn(model, layers, input_images[i])
        feature = hc.reshape(-1)
        desc_list.append(feature)

    desc_array = np.asarray(desc_list)
    desc_all = desc_array.astype(np.float64)

    Kpts_per_img = 1
    cluster_num = [128]
    for cc in range(len(cluster_num)):
        cluster_labels, cluster_centers = kmeans_hypercolumns(desc_all, cluster_num[cc])

        keypts_total = cluster_labels.shape[0]
        image_labels = np.zeros((keypts_total,))
        size_sum = 0
        size_cnt = 0
        for i in range(IMG_NUM):
            size_cnt = int(Kpts_per_img)
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
        nclusters = [math.ceil(cluster_num[cc]*0.05),math.ceil(cluster_num[cc]*0.1), math.ceil(cluster_num[cc]*0.25),
                     math.ceil(cluster_num[cc]*0.5), math.ceil(cluster_num[cc]*0.6), math.ceil(cluster_num[cc]*0.7),
                     math.ceil(cluster_num[cc]*0.8), math.ceil(cluster_num[cc]*0.9)]

        TPR = []
        FPR = []

        # arr_TP = []
        # arr_FP = []
        # arr_FN = []
        # arr_TN = []
        for n_c in range(len(nclusters)):
            nc = int(nclusters[n_c])
            TP = []
            FP = []
            for tao in np.arange(0.0, 1.1, 0.1):
                flag = np.zeros((IMG_NUM,))
                for i in range(IMG_NUM):
                    sel_pos = []
                    for j in range(nc):
                        img_ind = indices(sorted_clusters[j]['imageInd'], lambda x: x == i)
                        for cnt in range(len(img_ind)):
                            flag[cnt] = 1
                            sel_pos.append(sorted_clusters[j]['keyptInd'][0][img_ind[cnt]])

                    Inlier_per_img = len(sel_pos)
                    if (Inlier_per_img*1.0/Kpts_per_img) >= tao:
                        flag[i] = 1

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
                fn = positives_num - tp
                tn = negatives_num - fp
                # arr_TP.append(tp)
                # arr_FP.append(fp)
                # arr_FN.append(fn)
                # arr_TN.append(tn)
                TP.append(tp*1.0/positives_num)
                FP.append(fp*1.0/negatives_num)
            TPR.append(TP)
            FPR.append(FP)
        # with open(save_file, "w") as f:
        #     output = csv.writer(f)
        #     output.writerows([arr_TP, arr_FP, arr_FN, arr_TN])

        # plot ROC curve
        TPR = np.asarray(TPR)
        FPR = np.asarray(FPR)

        fig = plt.figure()
        plt.plot(FPR[0,:], TPR[0,:], linestyle='-', color='r', label='5% cluters')
        plt.plot(FPR[1,:], TPR[1,:], linestyle='-', color='g', label='10% clusters')
        plt.plot(FPR[2,:], TPR[2,:], linestyle='-', color='b', label='25% clusters')
        plt.plot(FPR[3,:], TPR[3,:], linestyle='-', color='c', label='50% clusters')
        plt.plot(FPR[4,:], TPR[4,:], linestyle='-', color='m', label='60% clusters')
        plt.plot(FPR[5,:], TPR[5,:], linestyle='-', color='y', label='70% clusters')
        plt.plot(FPR[6,:], TPR[6,:], linestyle='-', color='k', label='80% clusters')
        plt.plot(FPR[7,:], TPR[7,:], linestyle='-', color='0.75', label='90% clusters')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.title('ROC')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=4,  frameon=False, prop={'size': 9})
        plt.savefig(save_folder+'car_20_'+str(cluster_num[cc])+'.png')

