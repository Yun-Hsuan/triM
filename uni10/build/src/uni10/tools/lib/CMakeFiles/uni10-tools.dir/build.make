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
include src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/depend.make

# Include the progress variables for this target.
include src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/progress.make

# Include the compile flags for this target's objects.
include src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/flags.make

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/flags.make
src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o: ../src/uni10/tools/lib/uni10_tools.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/tools/lib/uni10_tools.cpp

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/uni10-tools.dir/uni10_tools.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/tools/lib/uni10_tools.cpp > CMakeFiles/uni10-tools.dir/uni10_tools.cpp.i

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/uni10-tools.dir/uni10_tools.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/tools/lib/uni10_tools.cpp -o CMakeFiles/uni10-tools.dir/uni10_tools.cpp.s

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o.requires:
.PHONY : src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o.requires

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o.provides: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o.requires
	$(MAKE) -f src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/build.make src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o.provides.build
.PHONY : src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o.provides

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o.provides.build: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/flags.make
src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o: ../src/uni10/tools/lib/uni10_tools_cpu.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o -c /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/tools/lib/uni10_tools_cpu.cpp

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.i"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/tools/lib/uni10_tools_cpu.cpp > CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.i

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.s"
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib && /opt/intel/composer_xe_2015.3.187/bin/intel64/icpc  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/tools/lib/uni10_tools_cpu.cpp -o CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.s

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o.requires:
.PHONY : src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o.requires

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o.provides: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o.requires
	$(MAKE) -f src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/build.make src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o.provides.build
.PHONY : src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o.provides

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o.provides.build: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o

uni10-tools: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o
uni10-tools: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o
uni10-tools: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/build.make
.PHONY : uni10-tools

# Rule to build all files generated by this target.
src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/build: uni10-tools
.PHONY : src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/build

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/requires: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools.cpp.o.requires
src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/requires: src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/uni10_tools_cpu.cpp.o.requires
.PHONY : src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/requires

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/clean:
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib && $(CMAKE_COMMAND) -P CMakeFiles/uni10-tools.dir/cmake_clean.cmake
.PHONY : src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/clean

src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/depend:
	cd /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/Yun-Hsuan/GitRepo/tensorlib/uni10 /home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10/tools/lib /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib /home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/uni10/tools/lib/CMakeFiles/uni10-tools.dir/depend
