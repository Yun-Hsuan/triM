# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/Yun-Hsuan/GitRepo/tensorlib/uni10

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/egB1.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/egB1.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/egB1.dir/flags.make

examples/CMakeFiles/egB1.dir/egB1.cpp.o: examples/CMakeFiles/egB1.dir/flags.make
examples/CMakeFiles/egB1.dir/egB1.cpp.o: ../examples/egB1.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/CMakeFiles/egB1.dir/egB1.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/examples && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/egB1.dir/egB1.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/examples/egB1.cpp

examples/CMakeFiles/egB1.dir/egB1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/egB1.dir/egB1.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/examples && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/examples/egB1.cpp > CMakeFiles/egB1.dir/egB1.cpp.i

examples/CMakeFiles/egB1.dir/egB1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/egB1.dir/egB1.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/examples && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/examples/egB1.cpp -o CMakeFiles/egB1.dir/egB1.cpp.s

examples/CMakeFiles/egB1.dir/egB1.cpp.o.requires:
.PHONY : examples/CMakeFiles/egB1.dir/egB1.cpp.o.requires

examples/CMakeFiles/egB1.dir/egB1.cpp.o.provides: examples/CMakeFiles/egB1.dir/egB1.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/egB1.dir/build.make examples/CMakeFiles/egB1.dir/egB1.cpp.o.provides.build
.PHONY : examples/CMakeFiles/egB1.dir/egB1.cpp.o.provides

examples/CMakeFiles/egB1.dir/egB1.cpp.o.provides.build: examples/CMakeFiles/egB1.dir/egB1.cpp.o

# Object files for target egB1
egB1_OBJECTS = \
"CMakeFiles/egB1.dir/egB1.cpp.o"

# External object files for target egB1
egB1_EXTERNAL_OBJECTS =

examples/egB1: examples/CMakeFiles/egB1.dir/egB1.cpp.o
examples/egB1: examples/CMakeFiles/egB1.dir/build.make
examples/egB1: libuni10.a
examples/egB1: /opt/intel/composer_xe_2015.3.187/mkl/lib/intel64/libmkl_rt.so
examples/egB1: /usr/lib64/libhdf5.so
examples/egB1: /usr/lib64/libz.so
examples/egB1: /usr/lib64/libdl.so
examples/egB1: /usr/lib64/libm.so
examples/egB1: /usr/lib64/libhdf5_cpp.so
examples/egB1: /usr/lib64/libhdf5.so
examples/egB1: /usr/lib64/libz.so
examples/egB1: /usr/lib64/libdl.so
examples/egB1: /usr/lib64/libm.so
examples/egB1: /usr/lib64/libhdf5_cpp.so
examples/egB1: examples/CMakeFiles/egB1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable egB1"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/egB1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/egB1.dir/build: examples/egB1
.PHONY : examples/CMakeFiles/egB1.dir/build

examples/CMakeFiles/egB1.dir/requires: examples/CMakeFiles/egB1.dir/egB1.cpp.o.requires
.PHONY : examples/CMakeFiles/egB1.dir/requires

examples/CMakeFiles/egB1.dir/clean:
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/egB1.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/egB1.dir/clean

examples/CMakeFiles/egB1.dir/depend:
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/Yun-Hsuan/GitRepo/tensorlib/uni10 /home/Yun-Hsuan/GitRepo/tensorlib/uni10/examples /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/examples /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/examples/CMakeFiles/egB1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/egB1.dir/depend

