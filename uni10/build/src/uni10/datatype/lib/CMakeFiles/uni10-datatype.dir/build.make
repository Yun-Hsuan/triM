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
include src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/depend.make

# Include the progress variables for this target.
include src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/progress.make

# Include the compile flags for this target's objects.
include src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/flags.make

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o: src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/flags.make
src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o: ../src/uni10/datatype/lib/Qnum.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/datatype/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/uni10-datatype.dir/Qnum.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/datatype/lib/Qnum.cpp

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/uni10-datatype.dir/Qnum.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/datatype/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/datatype/lib/Qnum.cpp > CMakeFiles/uni10-datatype.dir/Qnum.cpp.i

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/uni10-datatype.dir/Qnum.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/datatype/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/datatype/lib/Qnum.cpp -o CMakeFiles/uni10-datatype.dir/Qnum.cpp.s

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o.requires:
.PHONY : src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o.requires

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o.provides: src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o.requires
	$(MAKE) -f src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/build.make src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o.provides.build
.PHONY : src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o.provides

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o.provides.build: src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o

uni10-datatype: src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o
uni10-datatype: src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/build.make
.PHONY : uni10-datatype

# Rule to build all files generated by this target.
src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/build: uni10-datatype
.PHONY : src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/build

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/requires: src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/Qnum.cpp.o.requires
.PHONY : src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/requires

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/clean:
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/datatype/lib && $(CMAKE_COMMAND) -P CMakeFiles/uni10-datatype.dir/cmake_clean.cmake
.PHONY : src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/clean

src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/depend:
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/Yun-Hsuan/GitRepo/tensorlib/uni10 /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/datatype/lib /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/datatype/lib /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/uni10/datatype/lib/CMakeFiles/uni10-datatype.dir/depend
