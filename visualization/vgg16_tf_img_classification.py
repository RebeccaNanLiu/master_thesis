import cv2
import numpy as np
import tensorflow as tf
import os
import urllib2
import matplotlib.pyplot as plt

from datasets import imagenet
from datasets import dataset_utils
from nets import vgg
from preprocessing import vgg_preprocessing

# test image classification using VGG-16.

# before running the code:
# export PYTHONPATH=$PYTHONPATH:/Users/nanliu/models/slim

url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
checkpoints_dir = 'models/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)
    dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

slim = tf.contrib.slim

image_size = vgg.vgg_16.default_image_size

with tf.Graph().as_default():

    ## read images from url
    #url = 'https://upload.wikimedia.org/wikipedia/commons/d/d9/First_Student_IC_school_bus_202076.jpg'
    #image_string = urllib2.urlopen(url).read()
    #image = tf.image.decode_jpeg(image_string, channels=3)

    ## read images from local filefolder
    filenames = ['/Users/nanliu/PycharmProjects/vgg/data/madruga.jpg']
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)

    processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
                                             slim.get_model_variables('vgg_16'))

    with tf.Session() as sess:
        #op = sess.graph.get_operations()
        #for o in op:
        #    print o.name
        #    print o.values()

        init_fn(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        np_image, probabilities = sess.run([image, probabilities])

        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        coord.request_stop()
        coord.join(threads)

    plt.plot(probabilities.ravel())

    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        # Shift the index of a class name by one.
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index + 1]))

