## Random Walk method to detect outliers. Steps of using the code: 

1. cd ~/random_walk/build
2. make ..
3. make
4. ./random_walk net_prototxt_file net_caffemodel_file net_binaryproto_file synset_words_txt_file images_txt_file save_folder

example: ./random_walk /Users/nanliu/Documents/master_thesis/ResNet/ResNet-50-deploy.prototxt /Users/nanliu/Documents/master_thesis/ResNet/ResNet-50-model.caffemodel /Users/nanliu/Documents/master_thesis/ResNet/ResNet_mean.binaryproto /Users/nanliu/caffe-root/caffe/data/ilsvrc12/synset_words.txt /Users/nanliu/random_walk_resnet50/1/test.txt /Users/nanliu/random_walk_resnet50/1/test/