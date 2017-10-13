cmake_minimum_required(VERSION 2.8.8)

project(random_walk)


# opencv
find_package( OpenCV REQUIRED )

MESSAGE(STATUS "OpenCV INCLUDE DIRS ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV LINK LIBRARIES ${OpenCV_LIBS}")

# Caffe
find_package(Caffe REQUIRED)
SET(Caffe_DIR "/Users/nanliu/caffe-root/caffe/build")
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

MESSAGE(STATUS "Caffe INCLUDE DIRS ${Caffe_INCLUDE_DIRS}")
MESSAGE(STATUS "Caffe LINK LIBRARIES ${Caffe_LIBRARIES}")


include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Caffe_INCLUDE_DIRS}
)

#add_executable( img_classification src/img_classification.cpp)
#target_link_libraries( img_classification ${OpenCV_LIBS} ${Caffe_LIBRARIES})

add_executable( random_walk src/random_walk.cpp)
target_link_libraries( random_walk ${OpenCV_LIBS} ${Caffe_LIBRARIES})
