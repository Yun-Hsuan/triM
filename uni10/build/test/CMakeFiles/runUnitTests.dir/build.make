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
include test/CMakeFiles/runUnitTests.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/runUnitTests.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/runUnitTests.dir/flags.make

test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o: test/CMakeFiles/runUnitTests.dir/flags.make
test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o: ../test/testQnum.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/runUnitTests.dir/testQnum.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testQnum.cpp

test/CMakeFiles/runUnitTests.dir/testQnum.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runUnitTests.dir/testQnum.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testQnum.cpp > CMakeFiles/runUnitTests.dir/testQnum.cpp.i

test/CMakeFiles/runUnitTests.dir/testQnum.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runUnitTests.dir/testQnum.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testQnum.cpp -o CMakeFiles/runUnitTests.dir/testQnum.cpp.s

test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o.requires:
.PHONY : test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o.requires

test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o.provides: test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/runUnitTests.dir/build.make test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o.provides.build
.PHONY : test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o.provides

test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o.provides.build: test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o

test/CMakeFiles/runUnitTests.dir/testBond.cpp.o: test/CMakeFiles/runUnitTests.dir/flags.make
test/CMakeFiles/runUnitTests.dir/testBond.cpp.o: ../test/testBond.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/runUnitTests.dir/testBond.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/runUnitTests.dir/testBond.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testBond.cpp

test/CMakeFiles/runUnitTests.dir/testBond.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runUnitTests.dir/testBond.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testBond.cpp > CMakeFiles/runUnitTests.dir/testBond.cpp.i

test/CMakeFiles/runUnitTests.dir/testBond.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runUnitTests.dir/testBond.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testBond.cpp -o CMakeFiles/runUnitTests.dir/testBond.cpp.s

test/CMakeFiles/runUnitTests.dir/testBond.cpp.o.requires:
.PHONY : test/CMakeFiles/runUnitTests.dir/testBond.cpp.o.requires

test/CMakeFiles/runUnitTests.dir/testBond.cpp.o.provides: test/CMakeFiles/runUnitTests.dir/testBond.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/runUnitTests.dir/build.make test/CMakeFiles/runUnitTests.dir/testBond.cpp.o.provides.build
.PHONY : test/CMakeFiles/runUnitTests.dir/testBond.cpp.o.provides

test/CMakeFiles/runUnitTests.dir/testBond.cpp.o.provides.build: test/CMakeFiles/runUnitTests.dir/testBond.cpp.o

test/CMakeFiles/runUnitTests.dir/testTools.cpp.o: test/CMakeFiles/runUnitTests.dir/flags.make
test/CMakeFiles/runUnitTests.dir/testTools.cpp.o: ../test/testTools.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/runUnitTests.dir/testTools.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/runUnitTests.dir/testTools.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testTools.cpp

test/CMakeFiles/runUnitTests.dir/testTools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runUnitTests.dir/testTools.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testTools.cpp > CMakeFiles/runUnitTests.dir/testTools.cpp.i

test/CMakeFiles/runUnitTests.dir/testTools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runUnitTests.dir/testTools.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testTools.cpp -o CMakeFiles/runUnitTests.dir/testTools.cpp.s

test/CMakeFiles/runUnitTests.dir/testTools.cpp.o.requires:
.PHONY : test/CMakeFiles/runUnitTests.dir/testTools.cpp.o.requires

test/CMakeFiles/runUnitTests.dir/testTools.cpp.o.provides: test/CMakeFiles/runUnitTests.dir/testTools.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/runUnitTests.dir/build.make test/CMakeFiles/runUnitTests.dir/testTools.cpp.o.provides.build
.PHONY : test/CMakeFiles/runUnitTests.dir/testTools.cpp.o.provides

test/CMakeFiles/runUnitTests.dir/testTools.cpp.o.provides.build: test/CMakeFiles/runUnitTests.dir/testTools.cpp.o

test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o: test/CMakeFiles/runUnitTests.dir/flags.make
test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o: ../test/testMatrix.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/runUnitTests.dir/testMatrix.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testMatrix.cpp

test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runUnitTests.dir/testMatrix.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testMatrix.cpp > CMakeFiles/runUnitTests.dir/testMatrix.cpp.i

test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runUnitTests.dir/testMatrix.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testMatrix.cpp -o CMakeFiles/runUnitTests.dir/testMatrix.cpp.s

test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o.requires:
.PHONY : test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o.requires

test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o.provides: test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/runUnitTests.dir/build.make test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o.provides.build
.PHONY : test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o.provides

test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o.provides.build: test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o

test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o: test/CMakeFiles/runUnitTests.dir/flags.make
test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o: ../test/testUniTensor.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testUniTensor.cpp

test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runUnitTests.dir/testUniTensor.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testUniTensor.cpp > CMakeFiles/runUnitTests.dir/testUniTensor.cpp.i

test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runUnitTests.dir/testUniTensor.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testUniTensor.cpp -o CMakeFiles/runUnitTests.dir/testUniTensor.cpp.s

test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o.requires:
.PHONY : test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o.requires

test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o.provides: test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/runUnitTests.dir/build.make test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o.provides.build
.PHONY : test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o.provides

test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o.provides.build: test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o

test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o: test/CMakeFiles/runUnitTests.dir/flags.make
test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o: ../test/testNetwork.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/runUnitTests.dir/testNetwork.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testNetwork.cpp

test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runUnitTests.dir/testNetwork.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testNetwork.cpp > CMakeFiles/runUnitTests.dir/testNetwork.cpp.i

test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runUnitTests.dir/testNetwork.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test/testNetwork.cpp -o CMakeFiles/runUnitTests.dir/testNetwork.cpp.s

test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o.requires:
.PHONY : test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o.requires

test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o.provides: test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o.requires
	$(MAKE) -f test/CMakeFiles/runUnitTests.dir/build.make test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o.provides.build
.PHONY : test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o.provides

test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o.provides.build: test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o

# Object files for target runUnitTests
runUnitTests_OBJECTS = \
"CMakeFiles/runUnitTests.dir/testQnum.cpp.o" \
"CMakeFiles/runUnitTests.dir/testBond.cpp.o" \
"CMakeFiles/runUnitTests.dir/testTools.cpp.o" \
"CMakeFiles/runUnitTests.dir/testMatrix.cpp.o" \
"CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o" \
"CMakeFiles/runUnitTests.dir/testNetwork.cpp.o"

# External object files for target runUnitTests
runUnitTests_EXTERNAL_OBJECTS =

test/runUnitTests: test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o
test/runUnitTests: test/CMakeFiles/runUnitTests.dir/testBond.cpp.o
test/runUnitTests: test/CMakeFiles/runUnitTests.dir/testTools.cpp.o
test/runUnitTests: test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o
test/runUnitTests: test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o
test/runUnitTests: test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o
test/runUnitTests: test/CMakeFiles/runUnitTests.dir/build.make
test/runUnitTests: gtest-1.7.0/libgtest.a
test/runUnitTests: gtest-1.7.0/libgtest_main.a
test/runUnitTests: libuni10.so.1.0.0
test/runUnitTests: gtest-1.7.0/libgtest.a
test/runUnitTests: /opt/intel/composer_xe_2015.3.187/mkl/lib/intel64/libmkl_rt.so
test/runUnitTests: /usr/lib64/libhdf5.so
test/runUnitTests: /usr/lib64/libz.so
test/runUnitTests: /usr/lib64/libdl.so
test/runUnitTests: /usr/lib64/libm.so
test/runUnitTests: /usr/lib64/libhdf5_cpp.so
test/runUnitTests: /usr/lib64/libhdf5.so
test/runUnitTests: /usr/lib64/libz.so
test/runUnitTests: /usr/lib64/libdl.so
test/runUnitTests: /usr/lib64/libm.so
test/runUnitTests: /usr/lib64/libhdf5_cpp.so
test/runUnitTests: test/CMakeFiles/runUnitTests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable runUnitTests"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/runUnitTests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/runUnitTests.dir/build: test/runUnitTests
.PHONY : test/CMakeFiles/runUnitTests.dir/build

test/CMakeFiles/runUnitTests.dir/requires: test/CMakeFiles/runUnitTests.dir/testQnum.cpp.o.requires
test/CMakeFiles/runUnitTests.dir/requires: test/CMakeFiles/runUnitTests.dir/testBond.cpp.o.requires
test/CMakeFiles/runUnitTests.dir/requires: test/CMakeFiles/runUnitTests.dir/testTools.cpp.o.requires
test/CMakeFiles/runUnitTests.dir/requires: test/CMakeFiles/runUnitTests.dir/testMatrix.cpp.o.requires
test/CMakeFiles/runUnitTests.dir/requires: test/CMakeFiles/runUnitTests.dir/testUniTensor.cpp.o.requires
test/CMakeFiles/runUnitTests.dir/requires: test/CMakeFiles/runUnitTests.dir/testNetwork.cpp.o.requires
.PHONY : test/CMakeFiles/runUnitTests.dir/requires

test/CMakeFiles/runUnitTests.dir/clean:
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test && $(CMAKE_COMMAND) -P CMakeFiles/runUnitTests.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/runUnitTests.dir/clean

test/CMakeFiles/runUnitTests.dir/depend:
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/Yun-Hsuan/GitRepo/tensorlib/uni10 /home/Yun-Hsuan/GitRepo/tensorlib/uni10/test /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/test/CMakeFiles/runUnitTests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/runUnitTests.dir/depend

