# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


SET(CPACK_BINARY_BUNDLE "")
SET(CPACK_BINARY_CYGWIN "")
SET(CPACK_BINARY_DEB "")
SET(CPACK_BINARY_DRAGNDROP "")
SET(CPACK_BINARY_NSIS "")
SET(CPACK_BINARY_OSXX11 "")
SET(CPACK_BINARY_PACKAGEMAKER "")
SET(CPACK_BINARY_RPM "")
SET(CPACK_BINARY_STGZ "")
SET(CPACK_BINARY_TBZ2 "")
SET(CPACK_BINARY_TGZ "")
SET(CPACK_BINARY_TZ "")
SET(CPACK_BINARY_WIX "")
SET(CPACK_BINARY_ZIP "")
SET(CPACK_CMAKE_GENERATOR "Unix Makefiles")
SET(CPACK_COMPONENTS_ALL "libraries;headers;python_examples;examples;pyUni10;common;documentation")
SET(CPACK_COMPONENTS_ALL_SET_BY_USER "TRUE")
SET(CPACK_COMPONENT_COMMON_GROUP "Development")
SET(CPACK_COMPONENT_DOCUMENTATION_DISPLAYNAME "Uni10 API Reference")
SET(CPACK_COMPONENT_DOCUMENTATION_GROUP "Development")
SET(CPACK_COMPONENT_EXAMPLES_DISPLAYNAME "Uni10 C++ Examples")
SET(CPACK_COMPONENT_EXAMPLES_GROUP "Applications")
SET(CPACK_COMPONENT_GROUP_DEVELOPMENT_DESCRIPTION "Tools for developing Uni10 applications.")
SET(CPACK_COMPONENT_HEADERS_DEPENDS "libraries")
SET(CPACK_COMPONENT_HEADERS_DISPLAYNAME "C++ Headers")
SET(CPACK_COMPONENT_HEADERS_GROUP "Development")
SET(CPACK_COMPONENT_LIBRARIES_DISPLAYNAME "Uni10 Libraries")
SET(CPACK_COMPONENT_LIBRARIES_GROUP "Development")
SET(CPACK_COMPONENT_PYTHON_EXAMPLES_DEPENDS "pyUni10")
SET(CPACK_COMPONENT_PYTHON_EXAMPLES_DISPLAYNAME "pyUni10 Examples")
SET(CPACK_COMPONENT_PYTHON_EXAMPLES_GROUP "Applications")
SET(CPACK_COMPONENT_PYUNI10_DISPLAYNAME "pyUni10: Python Wrappers")
SET(CPACK_COMPONENT_PYUNI10_GROUP "Development")
SET(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
SET(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
SET(CPACK_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION "1")
SET(CPACK_GENERATOR "TGZ;RPM")
SET(CPACK_INCLUDE_TOPLEVEL_DIRECTORY "ON")
SET(CPACK_INSTALL_CMAKE_PROJECTS "/home/Yun-Hsuan/GitRepo/tensorlib/uni10/build;uni10;ALL;/")
SET(CPACK_INSTALL_PREFIX "/usr/local/uni10")
SET(CPACK_MODULE_PATH "/home/Yun-Hsuan/GitRepo/tensorlib/uni10/cmake/Modules/")
SET(CPACK_NSIS_DISPLAY_NAME "uni10")
SET(CPACK_NSIS_INSTALLER_ICON_CODE "")
SET(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
SET(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
SET(CPACK_NSIS_PACKAGE_NAME "uni10")
SET(CPACK_OUTPUT_CONFIG_FILE "/home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CPackConfig.cmake")
SET(CPACK_PACKAGE_DEFAULT_LOCATION "/")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "/home/Yun-Hsuan/GitRepo/tensorlib/uni10/INSTALL")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Uni10: the Universal Tensor Network Library")
SET(CPACK_PACKAGE_EXECUTABLES "Uni10;Uni10")
SET(CPACK_PACKAGE_FILE_NAME "uni10-1.0.0-Linux")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "uni10")
SET(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "uni10")
SET(CPACK_PACKAGE_NAME "uni10")
SET(CPACK_PACKAGE_RELOCATABLE "true")
SET(CPACK_PACKAGE_VENDOR "Uni10")
SET(CPACK_PACKAGE_VERSION "1.0.0")
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_RESOURCE_FILE_LICENSE "/home/Yun-Hsuan/GitRepo/tensorlib/uni10/GPL")
SET(CPACK_RESOURCE_FILE_README "/home/Yun-Hsuan/GitRepo/tensorlib/uni10/README.md")
SET(CPACK_RESOURCE_FILE_WELCOME "/usr/local/share/cmake-2.8/Templates/CPack.GenericWelcome.txt")
SET(CPACK_SET_DESTDIR "OFF")
SET(CPACK_SOURCE_CYGWIN "")
SET(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_SOURCE_IGNORE_FILES "/\\.git/;/build/;/debug/;/dep/;copyright.*;.*\\.sh")
SET(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/Yun-Hsuan/GitRepo/tensorlib/uni10/build/CPackSourceConfig.cmake")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "uni10-1.0.0")
SET(CPACK_SOURCE_STRIP_FILES "1")
SET(CPACK_SOURCE_TBZ2 "")
SET(CPACK_SOURCE_TGZ "")
SET(CPACK_SOURCE_TZ "")
SET(CPACK_SOURCE_ZIP "")
SET(CPACK_STRIP_FILES "")
SET(CPACK_SYSTEM_NAME "Linux")
SET(CPACK_TOPLEVEL_TAG "Linux")
SET(CPACK_WIX_SIZEOF_VOID_P "8")
