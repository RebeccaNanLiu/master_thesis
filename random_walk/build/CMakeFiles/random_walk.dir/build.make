# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.8.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.8.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nanliu/random_walk

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nanliu/random_walk/build

# Include any dependencies generated for this target.
include CMakeFiles/random_walk.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/random_walk.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/random_walk.dir/flags.make

CMakeFiles/random_walk.dir/src/random_walk.cpp.o: CMakeFiles/random_walk.dir/flags.make
CMakeFiles/random_walk.dir/src/random_walk.cpp.o: ../src/random_walk.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nanliu/random_walk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/random_walk.dir/src/random_walk.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/random_walk.dir/src/random_walk.cpp.o -c /Users/nanliu/random_walk/src/random_walk.cpp

CMakeFiles/random_walk.dir/src/random_walk.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/random_walk.dir/src/random_walk.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nanliu/random_walk/src/random_walk.cpp > CMakeFiles/random_walk.dir/src/random_walk.cpp.i

CMakeFiles/random_walk.dir/src/random_walk.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/random_walk.dir/src/random_walk.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nanliu/random_walk/src/random_walk.cpp -o CMakeFiles/random_walk.dir/src/random_walk.cpp.s

CMakeFiles/random_walk.dir/src/random_walk.cpp.o.requires:

.PHONY : CMakeFiles/random_walk.dir/src/random_walk.cpp.o.requires

CMakeFiles/random_walk.dir/src/random_walk.cpp.o.provides: CMakeFiles/random_walk.dir/src/random_walk.cpp.o.requires
	$(MAKE) -f CMakeFiles/random_walk.dir/build.make CMakeFiles/random_walk.dir/src/random_walk.cpp.o.provides.build
.PHONY : CMakeFiles/random_walk.dir/src/random_walk.cpp.o.provides

CMakeFiles/random_walk.dir/src/random_walk.cpp.o.provides.build: CMakeFiles/random_walk.dir/src/random_walk.cpp.o


# Object files for target random_walk
random_walk_OBJECTS = \
"CMakeFiles/random_walk.dir/src/random_walk.cpp.o"

# External object files for target random_walk
random_walk_EXTERNAL_OBJECTS =

random_walk: CMakeFiles/random_walk.dir/src/random_walk.cpp.o
random_walk: CMakeFiles/random_walk.dir/build.make
random_walk: /usr/local/lib/libopencv_stitching.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_superres.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_videostab.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_aruco.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_bgsegm.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_bioinspired.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_ccalib.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_dpm.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_fuzzy.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_hdf.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_line_descriptor.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_optflow.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_reg.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_saliency.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_stereo.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_structured_light.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_surface_matching.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_tracking.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_xfeatures2d.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_ximgproc.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_xobjdetect.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_xphoto.3.2.0.dylib
random_walk: /Users/nanliu/caffe-root/caffe/build/lib/libcaffe.1.0.0-rc4.dylib
random_walk: /usr/local/lib/libopencv_shape.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_phase_unwrapping.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_rgbd.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_calib3d.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_video.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_datasets.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_dnn.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_face.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_plot.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_text.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_features2d.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_flann.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_objdetect.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_ml.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_photo.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_highgui.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_videoio.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_imgcodecs.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_imgproc.3.2.0.dylib
random_walk: /usr/local/lib/libopencv_core.3.2.0.dylib
random_walk: /Users/nanliu/caffe-root/caffe/build/lib/libproto.a
random_walk: /usr/local/lib/libboost_system-mt.dylib
random_walk: /usr/local/lib/libboost_thread-mt.dylib
random_walk: /usr/local/lib/libboost_filesystem-mt.dylib
random_walk: /usr/local/lib/libboost_chrono-mt.dylib
random_walk: /usr/local/lib/libboost_date_time-mt.dylib
random_walk: /usr/local/lib/libboost_atomic-mt.dylib
random_walk: /usr/local/lib/libglog.dylib
random_walk: /usr/local/lib/libgflags.dylib
random_walk: /usr/local/lib/libprotobuf.dylib
random_walk: /usr/local/lib/libhdf5_cpp.dylib
random_walk: /usr/local/lib/libhdf5.dylib
random_walk: /usr/lib/libpthread.dylib
random_walk: /usr/lib/libz.dylib
random_walk: /usr/lib/libdl.dylib
random_walk: /usr/lib/libm.dylib
random_walk: /usr/local/lib/libhdf5_hl_cpp.dylib
random_walk: /usr/local/lib/libhdf5_hl.dylib
random_walk: /usr/local/lib/liblmdb.dylib
random_walk: /usr/local/lib/libleveldb.dylib
random_walk: /usr/local/lib/libsnappy.dylib
random_walk: /Users/nanliu/anaconda2/lib/libpython2.7.dylib
random_walk: /usr/local/lib/libboost_python-mt.dylib
random_walk: CMakeFiles/random_walk.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/nanliu/random_walk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable random_walk"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/random_walk.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/random_walk.dir/build: random_walk

.PHONY : CMakeFiles/random_walk.dir/build

CMakeFiles/random_walk.dir/requires: CMakeFiles/random_walk.dir/src/random_walk.cpp.o.requires

.PHONY : CMakeFiles/random_walk.dir/requires

CMakeFiles/random_walk.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/random_walk.dir/cmake_clean.cmake
.PHONY : CMakeFiles/random_walk.dir/clean

CMakeFiles/random_walk.dir/depend:
	cd /Users/nanliu/random_walk/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nanliu/random_walk /Users/nanliu/random_walk /Users/nanliu/random_walk/build /Users/nanliu/random_walk/build /Users/nanliu/random_walk/build/CMakeFiles/random_walk.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/random_walk.dir/depend

