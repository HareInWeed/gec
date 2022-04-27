cmake_minimum_required(VERSION 3.15)

project(
    gec
    VERSION 0.1.0
    DESCRIPTION "Generic Elliptic Curve Lib"
    LANGUAGES CXX)

option(GEC_ENABLE_CUDA "use CUDA" OFF)
if(GEC_ENABLE_CUDA)
    enable_language(CUDA)

    include(CheckLanguage)
    check_language(CUDA)
endif()

# Only do these if this is the main project, and not if it is included through
# `add_subdirectory`
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

# # force MSVC compile with utf-8
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
endif()

add_library(gec INTERFACE)
target_include_directories(gec INTERFACE "./include")
target_compile_features(gec INTERFACE cxx_std_14)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
