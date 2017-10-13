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

from skimage import io
from shutil import copyfile
from bs4 import BeautifulSoup

# helper functions to read Pascal images and labels
# helper functions to process the results
# helper functions to draw figures

root_dir = '/Users/nanliu/data/VOC2012/'
img_dir = os.path.join(root_dir, 'JPEGImages/')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')


def list_image_sets():
    """
    List all the image sets from Pascal VOC. Don't bother computing
    this on the fly, just remember it. It's faster.
    """
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


def imgs_from_category(cat_name, dataset):
    """
    Summary

    Args:
        cat_name (string): Category name as a string (from list_image_sets())
        dataset (string): "train", "val", "train_val", or "test" (if available)

    Returns:
        pandas dataframe: pandas DataFrame of all filenames from that category
    """
    filename = os.path.join(set_dir, cat_name + "_" + dataset + ".txt")
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=['filename', 'true'])
    return df


def imgs_from_category_as_list(cat_name, dataset):
    """
    Get a list of filenames for images in a particular category
    as a list rather than a pandas dataframe.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())
        dataset (string): "train", "val", "train_val", or "test" (if available)

    Returns:
        list of srings: all filenames from that category
    """
    df = imgs_from_category(cat_name, dataset)
    df = df[df['true'] == 1]
    return df['filename'].values


def annotation_file_from_img(img_name):
    """
    Given an image name, get the annotation file for that image

    Args:
        img_name (string): string of the image name, relative to
            the image directory.

    Returns:
        string: file path to the annotation file
    """
    return os.path.join(ann_dir, img_name) + '.xml'


def load_annotation(img_filename):
    """
    Load annotation file for a given image.

    Args:
        img_name (string): string of the image name, relative to
            the image directory.

    Returns:
        BeautifulSoup structure: the annotation labels loaded as a
            BeautifulSoup data structure
    """
    xml = ""
    with open(annotation_file_from_img(img_filename)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml)


# TODO: implement this
def get_all_obj_and_box(objname, img_set):
    img_list = imgs_from_category_as_list(objname, img_set)

    for img in img_list:
        annotation = load_annotation(img)


def load_img(img_filename):
    """
    Load image from the filename. Default is to load in color if
    possible.

    Args:
        img_name (string): string of the image name, relative to
            the image directory.

    Returns:
        np array of float32: an image as a numpy array of float32
    """
    img_filename = os.path.join(img_dir, img_filename + '.jpg')
    img = skimage.img_as_float(io.imread(
        img_filename)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def load_imgs(img_filenames):
    """
    Load a bunch of images from disk as np array.

    Args:
        img_filenames (list of strings): string of the image name, relative to
            the image directory.

    Returns:
        np array of float32: a numpy array of images. each image is
            a numpy array of float32
    """
    return np.array([load_img(fname) for fname in img_filenames])


def _load_data(category, data_type=None):
    """
    Loads all the data as a pandas DataFrame for a particular category.

    Args:
        category (string): Category name as a string (from list_image_sets())
        data_type (string, optional): "train" or "val"

    Raises:
        ValueError: when you don't give "train" or "val" as data_type

    Returns:
        pandas DataFrame: df of filenames and bounding boxes
    """
    if data_type is None:
        raise ValueError('Must provide data_type = train or val')
    to_find = category
    filename = os.path.join(root_dir, 'csvs/') + \
        data_type + '_' + \
        category + '.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        train_img_list = imgs_from_category_as_list(to_find, data_type)
        data = []
        for item in train_img_list:
            anno = load_annotation(item)
            objs = anno.findAll('object')
            for obj in objs:
                obj_names = obj.findChildren('name')
                for name_tag in obj_names:
                    if str(name_tag.contents[0]) == category:
                        fname = anno.findChild('filename').contents[0]
                        bbox = obj.findChildren('bndbox')[0]
                        xmin = int(bbox.findChildren('xmin')[0].contents[0])
                        ymin = int(bbox.findChildren('ymin')[0].contents[0])
                        xmax = int(bbox.findChildren('xmax')[0].contents[0])
                        ymax = int(bbox.findChildren('ymax')[0].contents[0])
                        data.append([fname, xmin, ymin, xmax, ymax])
        df = pd.DataFrame(
            data, columns=['fname', 'xmin', 'ymin', 'xmax', 'ymax'])
        df.to_csv(filename)
        return df


def get_image_url_list(category, data_type=None):
    """
    For a given data type, returns a list of filenames.

    Args:
        category (string): Category name as a string (from list_image_sets())
        data_type (string, optional): "train" or "val"

    Returns:
        list of strings: list of all filenames for that particular category
    """
    df = _load_data(category, data_type=data_type)
    image_url_list = list(
        unique_everseen(list(img_dir + df['fname'])))
    return image_url_list


def get_masks(cat_name, data_type, mask_type=None):
    """
    Return a list of masks for a given category and data_type.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())
        data_type (string, optional): "train" or "val"
        mask_type (string, optional): either "bbox1" or "bbox2" - whether to
            sum or add the masks for multiple objects

    Raises:
        ValueError: if mask_type is not valid

    Returns:
        list of np arrays: list of np arrays that are masks for the images
            in the particular category.
    """
    # change this to searching through the df
    # for the bboxes instead of relying on the order
    # so far, should be OK since I'm always loading
    # the df from disk anyway
    # mask_type should be bbox1 or bbox
    if mask_type is None:
        raise ValueError('Must provide mask_type')
    df = _load_data(cat_name, data_type=data_type)
    # load each image, turn into a binary mask
    masks = []
    prev_url = ""
    blank_img = None
    for row_num, entry in df.iterrows():
        img_url = os.path.join(img_dir, entry['fname'])
        if img_url != prev_url:
            if blank_img is not None:
                # TODO: options for how to process the masks
                # make sure the mask is from 0 to 1
                max_val = blank_img.max()
                if max_val > 0:
                    min_val = blank_img.min()
                    # print "min val before normalizing: ", min_val
                    # start at zero
                    blank_img -= min_val
                    # print "max val before normalizing: ", max_val
                    # max val at 1
                    blank_img /= max_val
                masks.append(blank_img)
            prev_url = img_url
            img = load_img(img_url)
            blank_img = np.zeros((img.shape[0], img.shape[1], 1))
        bbox = [entry['xmin'], entry['ymin'], entry['xmax'], entry['ymax']]
        if mask_type == 'bbox1':
            blank_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
        elif mask_type == 'bbox2':
            blank_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 1.0
        else:
            raise ValueError('Not a valid mask type')
    # TODO: options for how to process the masks
    # make sure the mask is from 0 to 1
    max_val = blank_img.max()
    if max_val > 0:
        min_val = blank_img.min()
        # print "min val before normalizing: ", min_val
        # start at zero
        blank_img -= min_val
        # print "max val before normalizing: ", max_val
        # max val at 1
        blank_img /= max_val
    masks.append(blank_img)
    return np.array(masks)


def get_imgs(cat_name, data_type=None):
    """
    Load and return all the images for a particular category.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())
        data_type (string, optional): "train" or "val"

    Returns:
        np array of images: np array of loaded images for the category
            and data_type.
    """
    image_url_list = get_image_url_list(cat_name, data_type=data_type)
    imgs = []
    for url in image_url_list:
        imgs.append(load_img(url))
    return np.array(imgs)


def display_image_and_mask(img, mask):
    """
    Display an image and it's mask side by side.

    Args:
        img (np array): the loaded image as a np array
        mask (np array): the loaded mask as a np array
    """
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(mask)
    ax2.set_title('Mask')
    plt.show(block=False)


def cat_name_to_cat_id(cat_name):
    """
    Transform a category name to an id number alphabetically.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())

    Returns:
        int: the integer that corresponds to the category name
    """
    cat_list = list_image_sets()
    cat_id_dict = dict(zip(cat_list, range(len(cat_list))))
    return cat_id_dict[cat_name]


def display_img_and_masks(
        img, true_mask, predicted_mask, block=False):
    """
    Display an image and it's two masks side by side.

    Args:
        img (np array): image as a np array
        true_mask (np array): true mask as a np array
        predicted_mask (np array): predicted_mask as a np array
        block (bool, optional): whether to display in a blocking manner or not.
            Default to False (non-blocking)
    """
    m_predicted_color = predicted_mask.reshape(
        predicted_mask.shape[0], predicted_mask.shape[1])
    m_true_color = true_mask.reshape(
        true_mask.shape[0], true_mask.shape[1])
    # m_predicted_color = predicted_mask
    # m_true_color = true_mask
    # plt.close(1)
    plt.figure(1)
    plt.clf()
    plt.axis('off')
    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, num=1)
    # f.clf()
    ax1.get_xaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax3.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax3.get_yaxis().set_ticks([])

    ax1.imshow(img)
    ax2.imshow(m_true_color)
    ax3.imshow(m_predicted_color)
    plt.draw()
    plt.show(block=block)


def load_data_multilabel(data_type=None):
    """
    Returns a data frame for all images in a given set in multilabel format.

    Args:
        data_type (string, optional): "train" or "val"

    Returns:
        pandas DataFrame: filenames in multilabel format
    """
    if data_type is None:
        raise ValueError('Must provide data_type = train or val')
    filename = os.path.join(set_dir, data_type + ".txt")
    cat_list = list_image_sets()
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=['filename'])
    # add all the blank rows for the multilabel case
    for cat_name in cat_list:
        df[cat_name] = 0
    for info in df.itertuples():
        index = info[0]
        fname = info[1]
        anno = load_annotation(fname)
        objs = anno.findAll('object')
        for obj in objs:
            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                tag_name = str(name_tag.contents[0])
                if tag_name in cat_list:
                    df.at[index, tag_name] = 1
    return df


# get the image list of all the categories
def get_imgs_from_label():
   class_list = list_image_sets()
   for a in class_list:
       name_list = imgs_from_category_as_list(a, 'train')
       category_file = open('/Users/nanliu/PycharmProjects/pascal-voc-python/result/train/' + a +'.txt', 'w')
       for item in name_list:
           category_file.write("%s\n" % item)
    return category_file


# copy the chosen category images into a different folder
def copy_from_to():
   names = open('/Users/nanliu/PycharmProjects/pascal-voc-python/result/val/aeroplane.txt', 'r')
   lines = names.read().splitlines()
   for item in lines:
       src = '/Users/nanliu/Documents/data/VOC2012/JPEGImages/'+item+'.jpg'
       dst = '/Users/nanliu/Documents/data/val/aeroplane/'+item+'.jpg'
       copyfile(src, dst)

# negative = total - positive
def get_negatives():
   name1 = open('/Users/nanliu/Desktop/randomwalk_aeroplane/positive.txt', 'r')
   name2 = open('/Users/nanliu/Desktop/randomwalk_aeroplane/total.txt', 'r')
   a = name1.read().splitlines()
   b = name2.read().splitlines()
   non_duplicates = [line for line in a if line not in b]
   non_duplicates += [line for line in b if line not in a]
   txtfile = open('/Users/nanliu/Desktop/randomwalk_aeroplane/negative.txt', 'w')
   for item in non_duplicates:
        txtfile.write("%s\n" % item)

# in each positive images get one bouding box for aeroplane
def getBND_pos(input, output, classname):
   names = open(input, 'r')
   lines = names.read().splitlines()
   text_file = open(output, "w")
   for item in lines:
       anno = load_annotation(item)
       objs = anno.findAll('object')
       cnt = 0
       for obj in objs:
           obj_names = obj.findChildren('name')
           for name_tag in obj_names:
               tag_name = str(name_tag.contents[0])
               if tag_name == classname:
                   cnt = cnt + 1
                   bbox = obj.findChildren('bndbox')[0]
                   xmin = int(bbox.findChildren('xmin')[0].contents[0])
                   ymin = int(bbox.findChildren('ymin')[0].contents[0])
                   xmax = int(bbox.findChildren('xmax')[0].contents[0])
                   ymax = int(bbox.findChildren('ymax')[0].contents[0])
                   if cnt == 1:
                       text_file.write("/Users/nanliu/data/VOC2012/JPEGImages/")
                       text_file.write("%s " % item)
                       text_file.write("%s " % xmin)
                       text_file.write("%s " % ymin)
                       text_file.write("%s " % xmax)
                       text_file.write("%s\n" % ymax)

# in each negative images get one bouding box
def getBND_neg(input, output):
   names = open(input, 'r')
   lines = names.read().splitlines()
   text_file = open(output, "w")
   for item in lines:
       anno = load_annotation(item)
       obj = anno.findAll('object')[0]
       obj_names = obj.findChildren('name')
       for name_tag in obj_names:
           tag_name = str(name_tag.contents[0])
           bbox = obj.findChildren('bndbox')[0]
           xmin = int(bbox.findChildren('xmin')[0].contents[0])
           ymin = int(bbox.findChildren('ymin')[0].contents[0])
           xmax = int(bbox.findChildren('xmax')[0].contents[0])
           ymax = int(bbox.findChildren('ymax')[0].contents[0])
           text_file.write("/Users/nanliu/data/VOC2012/JPEGImages/")
           text_file.write("%s " % item)
           text_file.write("%s " % xmin)
           text_file.write("%s " % ymin)
           text_file.write("%s " % xmax)
           text_file.write("%s\n" % ymax)

# delete same name in the list
def unique(input, output):
    s = set()
    with open(input, 'r') as f:
        for line in f:
            s.add(int(line.strip()))

    lSorted = sorted(s)

    with open(output, 'w') as f:
        for n in lSorted:
            f.write(str(n) + '\n')

def parseLatexAndWriteCSV(input, output):
    with open(input, 'r') as f:
        lines = f.readlines()

    content = ''.join([line.rstrip('\n') for line in lines])
    tables = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', content)

    i = 1
    for table in tables[:16]:
        with open(output + str(i) + '.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            rows = re.findall(r'&((?: *[0-9]+ *&){8})', table)
            for row in rows:
                formattedNumbers = []
                numbers = row.split('&')
                for number in numbers:
                    if number.strip():
                        formattedNumbers.append(number.strip())

                print(formattedNumbers)
                writer.writerow(formattedNumbers)

        i += 1

# get indices of negatives in the ranking list
def compareRankandNegatives(input1, input2):

    rankings = open(input1, 'r')
    rank_lines = rankings.read().splitlines()

    negatives = open(input2, 'r')
    neg_lines = negatives.read().splitlines()

    neg_rank = []
    counter = 0
    for item_r in rank_lines:
        counter = counter + 1
        for item_n in neg_lines:
            if item_r == item_n:
                neg_rank.append(counter)
    return neg_rank

def rounding(f):
    small = math.floor(f)
    big = math.ceil(f)
    if f-small < 0.5:
        return int(small)
    else:
        return int(big)

def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

# calculate TP and FP from the ranking list of images
def calc_tp_fp_from_ranking_list():
    rank_file = '/Users/nanliu/random_walk_resnet50/MITPlaces/attic/results.txt'
    neg_file = '/Users/nanliu/hypercolumns/MITPlaces/negatives/negatives.txt'
    save_file = '/Users/nanliu/random_walk_resnet50/MITPlaces/attic.txt'
    ## to get the total number of samples
    rank_list = open(rank_file, 'r')
    rank_lines = rank_list.read().splitlines()
    num_samples = int(len(rank_lines))
    GT_Pos = 500
    GT_Neg = num_samples - GT_Pos

    ## to get unique negatives postion in the ranking list
    neg_rank = compareRankandNegatives(rank_file, neg_file)
    neg_rank = list(sorted(set(neg_rank)))
    print neg_rank


    ## to calculate the disgard number
    Det_Neg = [rounding(num_samples*0.01), rounding(num_samples*0.02), rounding(num_samples*0.03), rounding(num_samples*0.04),
                   rounding(num_samples*0.05), rounding(num_samples*0.10), rounding(num_samples*0.15), rounding(num_samples*0.20)]
    Det_Pos = [num_samples-Det_Neg[i] for i in range(8)]


    ## to get tp, fp, fn, tn
    tp = []
    fp = []
    fn = []
    tn = []
    for i in range(8):
        ## to deal with the situation when len(neg_rank)>Det_Neg
        cont_ = int(len(neg_rank)) - Det_Neg[i]
        if cont_ > 0:
            neg_rank = neg_rank[cont_:]
        if Det_Pos[i] > neg_rank[-1]:
            fn_temp = Det_Neg[i]
            tn_temp = Det_Neg[i] - fn_temp
            tp_temp = GT_Pos - fn_temp
            fp_temp = GT_Neg - tn_temp
            tp.append(tp_temp)
            fp.append(fp_temp)
            fn.append(fn_temp)
            tn.append(tn_temp)
            print(tp_temp, fp_temp, fn_temp, tn_temp)
        else:
            inds = find(neg_rank, lambda x: x >= Det_Pos[i])

            tn_temp = int(len(inds))
            fn_temp = Det_Neg[i] - tn_temp
            fp_temp = GT_Neg - tn_temp
            tp_temp = GT_Pos - fn_temp
            tp.append(tp_temp)
            fp.append(fp_temp)
            fn.append(fn_temp)
            tn.append(tn_temp)
            print(tp_temp, fp_temp, fn_temp, tn_temp)

    with open(save_file, "w") as f:
        output = csv.writer(f)
        output.writerows([tp, fp, fn, tn])

# draw recall-precision curve
def draw_recall_precision(precision_RW, recall_RW, precision_LDA, recall_LDA):
    # Plot Precision-Recall curve
    fig = plt.figure()
    ax = plt.subplot(121)
    plt.plot(precision_RW[0], recall_RW[0], linestyle='-', marker='o', color='b', label='1% noise')
    plt.plot(precision_RW[4], recall_RW[4], linestyle='-', marker='o', color='m', label='5% noise')
    plt.plot(precision_RW[1], recall_RW[1], linestyle='-', marker='o', color='r', label='2% noise')
    plt.plot(precision_RW[5], recall_RW[5], linestyle='-', marker='o', color='y', label='10% noise')
    plt.plot(precision_RW[2], recall_RW[2], linestyle='-', marker='o', color='g', label='3% noise')
    plt.plot(precision_RW[6], recall_RW[6], linestyle='-', marker='o', color='k', label='15% noise')
    plt.plot(precision_RW[3], recall_RW[3], linestyle='-', marker='o', color='c', label='4% noise')
    plt.plot(precision_RW[7], recall_RW[7], linestyle='-', marker='o', color='0.75', label='20% noise')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.ylim([0.7, 1])
    plt.xlim([0.7, 1])
    plt.title('a) Random Walk Filtering')
    # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height])
    # ax.set_axisbelow(True)
    ax.xaxis.grid(color='0.55', linestyle='solid')
    ax.yaxis.grid(color='0.55', linestyle='solid')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, prop={'size': 7})

    ax = plt.subplot(122)
    plt.plot(precision_LDA[0], recall_LDA[0], linestyle='-', marker='o', color='b', label='1% noise')
    plt.plot(precision_LDA[4], recall_LDA[4], linestyle='-', marker='o', color='m', label='5% noise')
    plt.plot(precision_LDA[1], recall_LDA[1], linestyle='-', marker='o', color='r', label='2% noise')
    plt.plot(precision_LDA[5], recall_LDA[5], linestyle='-', marker='o', color='y', label='10% noise')
    plt.plot(precision_LDA[2], recall_LDA[2], linestyle='-', marker='o', color='g', label='3% noise')
    plt.plot(precision_LDA[6], recall_LDA[6], linestyle='-', marker='o', color='k', label='15% noise')
    plt.plot(precision_LDA[3], recall_LDA[3], linestyle='-', marker='o', color='c', label='4% noise')
    plt.plot(precision_LDA[7], recall_LDA[7], linestyle='-', marker='o', color='0.75', label='20% noise')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.ylim([0.7, 1])
    plt.xlim([0.7, 1])
    plt.title('b) E-LDA Filtering')
    # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width, box.height])
    # ax.set_axisbelow(True)
    ax.xaxis.grid(color='0.55', linestyle='solid')
    ax.yaxis.grid(color='0.55', linestyle='solid')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, prop={'size': 7})
    fig.savefig('/Users/nanliu/Desktop/test')
   # fig.savefig('/Users/nanliu/Desktop/test', bbox_inches='tight')
    plt.show()

# calculate recall and precision from the ranking list of the images
def calc_recall_precision(path):
    files = glob.glob(path)
    recall_1 = []
    precision_1 = []
    recall_2 = []
    precision_2 = []
    recall_3 = []
    precision_3 = []
    recall_4 = []
    precision_4 = []
    recall_5 = []
    precision_5 = []
    recall_6 = []
    precision_6 = []
    recall_7 = []
    precision_7 = []
    recall_8 = []
    precision_8 = []

    count = 0
    for file in files:
        count = count + 1
        flag = count % 8
        print file
        print count, flag
        f = open(file, 'r')
        lines = f.read().splitlines()

        tp_temp = lines[0].split(',')
        tp_ = [float(i) for i in tp_temp]

        fp_temp = lines[1].split(',')
        fp_ = [float(i) for i in fp_temp]

        fn_temp = lines[2].split(',')
        fn_ = [float(i) for i in fn_temp]

        tn_temp = lines[3].split(',')
        tn_ = [float(i) for i in tn_temp]
        if count < 9:
            for i in range(8):
                p = tp_[i]/(tp_[i]+fp_[i])
                r = tp_[i]/(tp_[i]+fn_[i])

                if flag == 1:
                    precision_1.append(p)
                    recall_1.append(r)
                if flag == 2:
                    precision_2.append(p)
                    recall_2.append(r)
                if flag == 3:
                    precision_3.append(p)
                    recall_3.append(r)
                if flag == 4:
                    precision_4.append(p)
                    recall_4.append(r)
                if flag == 5:
                    precision_5.append(p)
                    recall_5.append(r)
                if flag == 6:
                    precision_6.append(p)
                    recall_6.append(r)
                if flag == 7:
                    precision_7.append(p)
                    recall_7.append(r)
                if flag == 0:
                    precision_8.append(p)
                    recall_8.append(r)
        else:
            for i in range(8):
                p = tp_[i]/(tp_[i]+fp_[i])
                r = tp_[i]/(tp_[i]+fn_[i])

                if flag == 1:
                    precision_1[i] = precision_1[i] + p
                    recall_1[i] = recall_1[i] + r
                if flag == 2:
                    precision_2[i] = precision_2[i] + p
                    recall_2[i] = recall_2[i] + r
                if flag == 3:
                    precision_3[i] = precision_3[i] + p
                    recall_3[i] = recall_3[i] + r
                if flag == 4:
                    precision_4[i] = precision_4[i] + p
                    recall_4[i] = recall_4[i] + r
                if flag == 5:
                    precision_5[i] = precision_5[i] + p
                    recall_5[i] = recall_5[i] + r
                if flag == 6:
                    precision_6[i] = precision_6[i] + p
                    recall_6[i] = recall_6[i] + r
                if flag == 7:
                    precision_7[i] = precision_7[i] + p
                    recall_7[i] = recall_7[i] + r
                if flag == 0:
                    precision_8[i] = precision_8[i] + p
                    recall_8[i] = recall_8[i] + r
    class_num = 18.0
    precision_avg_1 = [i/class_num for i in precision_1]
    precision_avg_2 = [i/class_num for i in precision_2]
    precision_avg_3 = [i/class_num for i in precision_3]
    precision_avg_4 = [i/class_num for i in precision_4]
    precision_avg_5 = [i/class_num for i in precision_5]
    precision_avg_6 = [i/class_num for i in precision_6]
    precision_avg_7 = [i/class_num for i in precision_7]
    precision_avg_8 = [i/class_num for i in precision_8]
    recall_avg_1 = [i/class_num for i in recall_1]
    recall_avg_2 = [i/class_num for i in recall_2]
    recall_avg_3 = [i/class_num for i in recall_3]
    recall_avg_4 = [i/class_num for i in recall_4]
    recall_avg_5 = [i/class_num for i in recall_5]
    recall_avg_6 = [i/class_num for i in recall_6]
    recall_avg_7 = [i/class_num for i in recall_7]
    recall_avg_8 = [i/class_num for i in recall_8]

    precision_array = []
    recall_array = []
    precision_array.append(precision_avg_1)
    precision_array.append(precision_avg_2)
    precision_array.append(precision_avg_3)
    precision_array.append(precision_avg_4)
    precision_array.append(precision_avg_5)
    precision_array.append(precision_avg_6)
    precision_array.append(precision_avg_7)
    precision_array.append(precision_avg_8)
    recall_array.append(recall_avg_1)
    recall_array.append(recall_avg_2)
    recall_array.append(recall_avg_3)
    recall_array.append(recall_avg_4)
    recall_array.append(recall_avg_5)
    recall_array.append(recall_avg_6)
    recall_array.append(recall_avg_7)
    recall_array.append(recall_avg_8)

    return precision_array, recall_array


if __name__ == "__main__":
    calc_tp_fp_from_ranking_list

    # path_LDA = '/Users/nanliu/Documents/master_thesis/exemplar-LDA/csv_LDA/*.csv'
    # path_RW = '/Users/nanliu/random_walk_resnet50/rw_corrected_18/*.txt'
    # precision_LDA, recall_LDA = calc_recall_precision(path_LDA)
    # precision_RW, recall_RW = calc_recall_precision(path_RW)
    # draw_recall_precision(precision_RW, recall_RW, precision_LDA, recall_LDA)

    ## transfer pdf table to .csv
    # for table in list_image_sets():
    #     parseLatexAndWriteCSV('/Users/nanliu/Desktop/table_{0}/test.tex'.format(table), '/Users/nanliu/Desktop/tables/{0}_table_'.format(table))

    # table = 'aeroplane'
    # parseLatexAndWriteCSV('/Users/nanliu/Desktop/table_aeroplane/test.tex'.format(table), '/Users/nanliu/Desktop/tables/aeroplane_table_'.format(table))

    ## get the bonding boxes for images
    # getBND_pos('/Users/nanliu/Documents/data/result/positives/bicycle.txt',
    #            '/Users/nanliu/Documents/master_thesis/exemplar-LDA/LDA_bicycle/bnd_pos.txt', 'bicycle')
    # getBND_neg('/Users/nanliu/Documents/master_thesis/exemplar-LDA/LDA_bicycle/negative.txt',
    #            '/Users/nanliu/Documents/master_thesis/exemplar-LDA/LDA_bicycle/bnd_neg.txt')


    ## for Exemplar-LDA, get the ranking list from index
   # index_file = open('/Users/nanliu/Desktop/exemplar-LDA/LDA_chair/index_20.txt', 'r')
   # index_lines = index_file.read().splitlines()
   # total_file = open('/Users/nanliu/Desktop/exemplar-LDA/LDA_chair/total.txt', 'r')
   # total_lines = total_file.read().splitlines()
   # text_file = open('/Users/nanliu/Desktop/exemplar-LDA/LDA_chair/ranking_20.txt', 'w')
   # for item in index_lines:
   #     counter = 0
   #     for t in total_lines:
   #         counter = counter + 1
   #         if item == str(counter):
   #             text_file.write("%s\n" % t)
   #             break



