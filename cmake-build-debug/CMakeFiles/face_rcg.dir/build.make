# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /home/swfxliyiyu/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/171.4073.41/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/swfxliyiyu/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/171.4073.41/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/swfxliyiyu/CLionProjects/face_rcg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/face_rcg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/face_rcg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/face_rcg.dir/flags.make

CMakeFiles/face_rcg.dir/src/main.cpp.o: CMakeFiles/face_rcg.dir/flags.make
CMakeFiles/face_rcg.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/face_rcg.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/face_rcg.dir/src/main.cpp.o -c /home/swfxliyiyu/CLionProjects/face_rcg/src/main.cpp

CMakeFiles/face_rcg.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/face_rcg.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/swfxliyiyu/CLionProjects/face_rcg/src/main.cpp > CMakeFiles/face_rcg.dir/src/main.cpp.i

CMakeFiles/face_rcg.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/face_rcg.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/swfxliyiyu/CLionProjects/face_rcg/src/main.cpp -o CMakeFiles/face_rcg.dir/src/main.cpp.s

CMakeFiles/face_rcg.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/face_rcg.dir/src/main.cpp.o.requires

CMakeFiles/face_rcg.dir/src/main.cpp.o.provides: CMakeFiles/face_rcg.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/face_rcg.dir/build.make CMakeFiles/face_rcg.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/face_rcg.dir/src/main.cpp.o.provides

CMakeFiles/face_rcg.dir/src/main.cpp.o.provides.build: CMakeFiles/face_rcg.dir/src/main.cpp.o


CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o: CMakeFiles/face_rcg.dir/flags.make
CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o: ../src/face_features/detect_face.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o -c /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/detect_face.cpp

CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/detect_face.cpp > CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.i

CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/detect_face.cpp -o CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.s

CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o.requires:

.PHONY : CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o.requires

CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o.provides: CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o.requires
	$(MAKE) -f CMakeFiles/face_rcg.dir/build.make CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o.provides.build
.PHONY : CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o.provides

CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o.provides.build: CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o


CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o: CMakeFiles/face_rcg.dir/flags.make
CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o: ../src/face_features/load_data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o -c /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/load_data.cpp

CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/load_data.cpp > CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.i

CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/load_data.cpp -o CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.s

CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o.requires:

.PHONY : CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o.requires

CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o.provides: CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o.requires
	$(MAKE) -f CMakeFiles/face_rcg.dir/build.make CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o.provides.build
.PHONY : CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o.provides

CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o.provides.build: CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o


CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o: CMakeFiles/face_rcg.dir/flags.make
CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o: ../src/face_features/pre_treat.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o -c /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/pre_treat.cpp

CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/pre_treat.cpp > CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.i

CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/swfxliyiyu/CLionProjects/face_rcg/src/face_features/pre_treat.cpp -o CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.s

CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o.requires:

.PHONY : CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o.requires

CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o.provides: CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o.requires
	$(MAKE) -f CMakeFiles/face_rcg.dir/build.make CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o.provides.build
.PHONY : CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o.provides

CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o.provides.build: CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o


# Object files for target face_rcg
face_rcg_OBJECTS = \
"CMakeFiles/face_rcg.dir/src/main.cpp.o" \
"CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o" \
"CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o" \
"CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o"

# External object files for target face_rcg
face_rcg_EXTERNAL_OBJECTS =

face_rcg: CMakeFiles/face_rcg.dir/src/main.cpp.o
face_rcg: CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o
face_rcg: CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o
face_rcg: CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o
face_rcg: CMakeFiles/face_rcg.dir/build.make
face_rcg: /usr/local/lib/libopencv_ml.so.3.2.0
face_rcg: /usr/local/lib/libopencv_objdetect.so.3.2.0
face_rcg: /usr/local/lib/libopencv_shape.so.3.2.0
face_rcg: /usr/local/lib/libopencv_stitching.so.3.2.0
face_rcg: /usr/local/lib/libopencv_superres.so.3.2.0
face_rcg: /usr/local/lib/libopencv_videostab.so.3.2.0
face_rcg: /usr/local/lib/libopencv_calib3d.so.3.2.0
face_rcg: /usr/local/lib/libopencv_features2d.so.3.2.0
face_rcg: /usr/local/lib/libopencv_flann.so.3.2.0
face_rcg: /usr/local/lib/libopencv_highgui.so.3.2.0
face_rcg: /usr/local/lib/libopencv_photo.so.3.2.0
face_rcg: /usr/local/lib/libopencv_video.so.3.2.0
face_rcg: /usr/local/lib/libopencv_videoio.so.3.2.0
face_rcg: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
face_rcg: /usr/local/lib/libopencv_imgproc.so.3.2.0
face_rcg: /usr/local/lib/libopencv_core.so.3.2.0
face_rcg: CMakeFiles/face_rcg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable face_rcg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/face_rcg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/face_rcg.dir/build: face_rcg

.PHONY : CMakeFiles/face_rcg.dir/build

CMakeFiles/face_rcg.dir/requires: CMakeFiles/face_rcg.dir/src/main.cpp.o.requires
CMakeFiles/face_rcg.dir/requires: CMakeFiles/face_rcg.dir/src/face_features/detect_face.cpp.o.requires
CMakeFiles/face_rcg.dir/requires: CMakeFiles/face_rcg.dir/src/face_features/load_data.cpp.o.requires
CMakeFiles/face_rcg.dir/requires: CMakeFiles/face_rcg.dir/src/face_features/pre_treat.cpp.o.requires

.PHONY : CMakeFiles/face_rcg.dir/requires

CMakeFiles/face_rcg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/face_rcg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/face_rcg.dir/clean

CMakeFiles/face_rcg.dir/depend:
	cd /home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/swfxliyiyu/CLionProjects/face_rcg /home/swfxliyiyu/CLionProjects/face_rcg /home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug /home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug /home/swfxliyiyu/CLionProjects/face_rcg/cmake-build-debug/CMakeFiles/face_rcg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/face_rcg.dir/depend

