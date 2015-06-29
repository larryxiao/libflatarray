cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

# *0. Set the build target here
#set(INTEL_SSE TRUE)
#set(ARM_NEON TRUE)
#set(INTEL_AVX TRUE)
#set(INTEL_AVX512 TRUE)

if (INTEL_SSE)
    set(ADDITIONAL_COMPILE_FLAGS  "-D __SSE__")
endif (INTEL_SSE)

if (INTEL_AVX)
    set(ADDITIONAL_COMPILE_FLAGS  "-D __AVX__ -mavx")
endif (INTEL_AVX)

if (INTEL_AVX512)
    # [c++ - Appending compiler flags to a file with CMake - Stack Overflow](http://stackoverflow.com/questions/20244800/appending-compiler-flags-to-a-file-with-cmake)
    set(CMAKE_C_COMPILER "/home/walnut/Developer/libflatarray/Software/gcc_install_5.0.0-mpx-r214719/bin/x86_64-pc-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "/home/walnut/Developer/libflatarray/Software/gcc_install_5.0.0-mpx-r214719/bin/x86_64-pc-linux-gnu-c++")
    set(CMAKE_C_FLAGS "-B ~/Developer/libflatarray/Software/binutils-gdb_install_2.24.51.20140422/bin -L /usr/lib/x86_64-linux-gnu/ -L ~/Developer/libflatarray/Software/gcc_install_5.0.0-mpx-r214719/lib64")
    set(CMAKE_CXX_FLAGS "-B ~/Developer/libflatarray/Software/binutils-gdb_install_2.24.51.20140422/bin -L /usr/lib/x86_64-linux-gnu/ -L ~/Developer/libflatarray/Software/gcc_install_5.0.0-mpx-r214719/lib64")
    set(ADDITIONAL_COMPILE_FLAGS  "-D __AVX512__ -mavx512f")
    set(BOOST_LIBRARYDIR "/usr/lib64/")
    # test
    set(ADDITIONAL_COMPILE_FLAGS  "-D __AVX512__")
endif(INTEL_AVX512)

if (ARM_NEON)
    # Cross Compile
    set(CMAKE_C_COMPILER "arm-linux-gnueabi-gcc")
    set(CMAKE_CXX_COMPILER "arm-linux-gnueabi-g++")
    #set(ADDITIONAL_COMPILE_FLAGS "-D __NEON__ -march=armv7-a -mtune=cortex-a8 -mfpu=neon -mfloat-abi=softfp")
    set(CMAKE_C_FLAGS "-D __NEON__ -march=armv7-a -mtune=cortex-a8 -mfpu=neon -mfloat-abi=softfp")
    set(CMAKE_CXX_FLAGS "-D __NEON__ -march=armv7-a -mtune=cortex-a8 -mfpu=neon -mfloat-abi=softfp")
    set(Boost_NO_SYSTEM_PATHS TRUE)
    if (Boost_NO_SYSTEM_PATHS)
        set(BOOST_ROOT "~/Developer/libflatarray/Software/boost-install")
        set(BOOST_INCLUDEDIR "~/Developer/libflatarray/Software/boost-install/include")
        set(BOOST_LIBRARYDIR "~/Developer/libflatarray/Software/boost-install/lib")
    endif (Boost_NO_SYSTEM_PATHS)
endif (ARM_NEON)

# 1. GENERIC PACKAGE INFORMATION
set(PACKAGE_NAME libflatarray)
set(PACKAGE_VERSION "0.2.0")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http://www.libgeodecomp.org/libflatarray.html")
set(PACKAGE_EMAIL "users@libgeodecomp.org")
project(${PACKAGE_NAME})

find_package(Boost REQUIRED date_time filesystem system)
find_package(CUDA)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
if(SUPPORTS_MARCH_NATIVE AND NOT APPLE)
  set(DEFAULT_FLAGS "${DEFAULT_FLAGS} -march=native")
endif()

# pretty print build options
function(print_options)
  message("-- The following options have been configured:")
  message(${OPTIONS_LIST})
  message("")
endfunction(print_options)

# dumps selected build options to a config header
function(dump_config outfile)
  set(CONTENT "#ifndef LIBFLATARRAY_CONFIG_H\n\n")
  set(CONTENT "${CONTENT}${CONFIG_HEADER}\n")
  set(CONTENT "${CONTENT}#endif\n")
  file(WRITE "${outfile}.new" "${CONTENT}")

  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${CMAKE_CURRENT_SOURCE_DIR}/${outfile}" "${CMAKE_CURRENT_SOURCE_DIR}/${outfile}.new" RESULT_VARIABLE res)
  if(res GREATER 0)
    file(WRITE "${outfile}" "${CONTENT}")
  endif()

  file(REMOVE "${outfile}.new")
endfunction(dump_config)

# generic function to add user-configurable options. add_to_header may be used to propagate the option to a header file.
function(add_config_option name help_message default add_to_header)
  if(NOT DEFINED ${name})
    set(${name} "${default}")
    set(${name} "${default}" CACHE STRING "${help_message}" FORCE)
  endif()

  set(OPTIONS_LIST "${OPTIONS_LIST}\n\n * ${name}=\"${${name}}\",\n   default=\"${default}\"\n   ${help_message}" PARENT_SCOPE)

  if(add_to_header)
    if(${name})
      set(CONFIG_HEADER "${CONFIG_HEADER}#define LIBFLATARRAY_${name} ${${name}}\n" PARENT_SCOPE)
    endif()
  endif()
endfunction(add_config_option)

add_config_option(ADDITIONAL_COMPILE_FLAGS "Add these flags when compiling." "${DEFAULT_FLAGS}" false)
add_config_option(WITH_CUDA "Enable modules which harness Nvidia CUDA GPUs" ${CUDA_FOUND} true)
add_config_option(LIB_INSTALL_DIR "Directory where libraries will be installed to, may be overridden for 64-bit installations" "lib" false)

if(WITH_CUDA AND NOT CUDA_FOUND)
  message(FATAL_ERROR "WITH_CUDA selected, but could not find the NVIDIA CUDA toolkit.")
endif()

dump_config("src/config.h")
print_options()

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${ADDITIONAL_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_COMPILE_FLAGS}")

# 2. CONFIGURE INSTALLER
set(CPACK_PACKAGE_NAME ${PACKAGE_NAME})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION ${PACKAGE_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${PACKAGE_NAME})

# will be shown e.g. in windows' control center package info
set(CPACK_PACKAGE_VENDOR ${PACKAGE_VENDOR})
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${PACKAGE_NAME})

include(CPack)

set(ConfigPackageLocation ${LIB_INSTALL_DIR}/cmake/${PACKAGE_NAME})

if(CMAKE_VERSION VERSION_GREATER 2.8.10)
  include(CMakePackageConfigHelpers)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion)

  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}ConfigVersion.cmake"
    DESTINATION "${ConfigPackageLocation}")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  "
get_filename_component(LIBFLATARRAY_CMAKE_DIR \${CMAKE_CURRENT_LIST_FILE} PATH)
set(libflatarray_INCLUDE_DIR \"\${LIBFLATARRAY_CMAKE_DIR}/../../../include\")
")

set(ConfigPackageLocation ${LIB_INSTALL_DIR}/cmake/${PACKAGE_NAME})
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  DESTINATION "${ConfigPackageLocation}")



file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
foreach(i ${HEADERS})
  install(FILES ${i} DESTINATION include/${PACKAGE_NAME})
endforeach()

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/*.hpp")
foreach(i ${HEADERS})
  install(FILES ${i} DESTINATION include/${PACKAGE_NAME}/detail)
endforeach()

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/testbed/*.hpp")
foreach(i ${HEADERS})
  install(FILES ${i} DESTINATION include/${PACKAGE_NAME}/testbed)
endforeach()

# 3. BUILD DESCRIPTION
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(WITH_CUDA)
  set(CUDA_NVCC_FLAGS "--gpu-architecture=sm_20")
endif()

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_custom_target(check echo "Tests passed.")

add_subdirectory(test)
add_subdirectory(examples)
