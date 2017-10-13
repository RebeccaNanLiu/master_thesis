import re
import os
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as pp

# draw bar chart for analysis.

def readtxt(file):
    f = open(file, 'r')
    lines = f.read().splitlines()
    data = [int(i) for i in lines]

    return data

def draw_keypoints_number(file_1, file_2):
    hc = readtxt(file_1)
    sf = readtxt(file_2)

    hc = np.asarray(hc)
    sf = np.asarray(sf)

    N = 20
    ind = np.arange(20)  # the x locations for the groups
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, hc, width, color='r')
    rects2 = ax.bar(ind + width, sf, width, color='y')

    ax.set_title('Comparison of keypoints number between Hypercolumn and SIFT based method')
    ax.set_ylabel('Number of keypoints')
    ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10','C11', 'C12', 'C13', 'C14', 'C15',
    #                     'C16', 'C17', 'C18', 'C19', 'C20'))
    ax.set_xlabel('Codewords')
    ax.legend((rects1[0], rects2[0]), ('Hypercolumn', 'SIFT'))
    plt.show()

def draw_feature_distribution_chart():
    synthetic = [0.04, 0.04, 4, 0.04, 2, 0.04]
    pascal = [2, 3, 0.04, 8, 0.04, 2]
    mitplaces = [0.04, 1, 0.04, 3, 1, 0.04]

    N = 6
    ind = np.arange(6)  # the x locations for the groups
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, synthetic, width, color='c', label = 'Synthetic_v3')
    rects2 = ax.bar(ind + width, pascal, width, color='g', label = 'Pascal')
    rects3 = ax.bar(ind - width, mitplaces, width, color='b', label = 'MITPlaces')
    ax.set_title('Feature distribution on 3 datasets')
    ax.set_ylabel('Number of classes')
    ax.set_xticks(ind)
    ax.set_xticklabels(('blcok2_pool', 'block3_pool', 'block4_conv3', 'block4_pool', 'block5_conv3', 'block5_pool'))
    ax.set_xlabel('Hypercolumn Features')
    ax.legend()
    plt.show()

def draw_k_distribution():
    hypercolumn = [1,1,6,7,8,2,1,0]
    sift = [2,2,2,4,7,7,0,2]

    N = 8
    ind = np.arange(N)
    width = 0.1

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.bar(ind, hypercolumn, width, color='r', label='Hypercolumn')
    ax.bar(ind + width, sift, width, color='y', label='SIFT')
    ax.set_title('Distribution of k on 3 datasets')
    ax.set_ylabel('Number of classes')
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels(('512', '1024', '2048', '4096', '8192', '16384', '32768', '65536'))
    ax.set_xlabel('k')
    ax.legend()
    plt.show()



def draw_m_distribution():
    offset = 0.02
    synthetic_hypercolumn = [offset, offset, offset, offset, offset, offset, 4, 2]
    synthetic_sift = [offset, offset, offset, 4, 1, offset, offset, 1]

    pascal_hypercolumn = [5, 6, 1, 2, 1, offset, offset, offset]
    pascal_sift = [3, 1, 4, 2, 2, 2, 1, offset]

    mitplaces_hypercolumn = [2, 1, 1, 1, offset, offset, offset,offset]
    mitplaces_sift = [1, offset, offset, offset, 3, offset, 1, offset]

    N = 8
    ind = np.arange(8)
    width = 0.1

    fig = plt.figure()
    ax = plt.subplot(111)
    # ax.bar(ind, synthetic_hypercolumn, width, color='r', label='Hypercolumn')
    # ax.bar(ind + width, synthetic_sift, width, color='y', label='SIFT')
    # ax.set_title('Distribution of m on Synthetic_v3')
    # ax.set_ylabel('Number of classes')
    # ax.set_xticks(ind+width/2)
    # ax.set_xticklabels(('5%', '10%', '25%', '50%', '60%', '70%', '80%', '90%'))
    # ax.set_xlabel('m')
    # ax.legend()

    # ax = plt.subplots(132)
    # ax.bar(ind, pascal_hypercolumn, width, color='r', label='Hypercolumn')
    # ax.bar(ind + width, pascal_sift, width, color='y', label='SIFT')
    # ax.set_title('Distribution of m on Pascal')
    # ax.set_ylabel('Number of classes')
    # ax.set_xticks(ind+width/2)
    # ax.set_xticklabels(('5%', '10%', '25%', '50%', '60%', '70%', '80%', '90%'))
    # ax.set_xlabel('m')
    # ax.legend()
    #
    # ax = plt.subplots(133)
    ax.bar(ind, mitplaces_hypercolumn, width, color='r', label='Hypercolumn')
    ax.bar(ind + width, mitplaces_sift, width, color='y', label='SIFT')
    ax.set_title('Distribution of m on MITPlaces')
    ax.set_ylabel('Number of classes')
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels(('5%', '10%', '25%', '50%', '60%', '70%', '80%', '90%'))
    ax.set_xlabel('m')
    ax.legend()

    plt.show()



if __name__ == "__main__":
    # file_1 = '/Users/nanliu/hypercolumn-visualization/hypercolumn_hist.txt'
    # file_2 = '/Users/nanliu/hypercolumn-visualization/sift_hist.txt'
    # draw_keypoints_number(file_1, file_2)

    draw_k_distribution()