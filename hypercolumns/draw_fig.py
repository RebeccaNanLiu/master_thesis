import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
import csv
import sys
import re
import os
import glob
import errno

# Draw ROC for Hypercolumn and SIFT

def read_and_draw(file_1, file_2):
    f1 = open(file_1, 'r')
    lines = f1.read().splitlines()


    fp_temp = lines[8].split(',')
    fpr_hy = [float(i) for i in fp_temp]
    tp_temp = lines[9].split(',')
    tpr_hy = [float(i) for i in tp_temp]

    f2 = open(file_2, 'r')
    lines = f2.read().splitlines()
    fp_temp = lines[12].split(',')
    fpr_sift = [float(i) for i in fp_temp]
    tp_temp = lines[13].split(',')
    tpr_sift = [float(i) for i in tp_temp]

    fig = plt.figure()
    plt.plot(fpr_hy, tpr_hy, linestyle='-', linewidth=4.0, color='r', label='Hypercolumn')
    plt.plot(fpr_sift, tpr_sift, linestyle='-', linewidth=4.0, color='b', label='SIFT')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, prop={'size': 9})
    plt.savefig('/Users/nanliu/Desktop/v90' + '.png')
    plt.show()


if __name__ == "__main__":
    file_1 = '/Users/nanliu/hypercolumns/synthetic_results/roc_txts/v90_conv13_2048.txt'
    file_2 = '/Users/nanliu/hypercolumns/synthetic_results/roc_txts/v90_sift_2048.txt'
    read_and_draw(file_1, file_2)