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

add_library(gec INTERFACE)
target_include_directories(gec INTERFACE "./include")
target_compile_features(gec INTERFACE cxx_std_14)

option(GEC_ENABLE_AVX2 "use avx2" OFF)
if(GEC_ENABLE_AVX2)
    target_compile_definitions(gec INTERFACE GEC_ENABLE_AVX2)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(
            gec INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>"
                          "$<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>")
    else()
        target_compile_options(
            gec INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>"
                          "$<$<COMPILE_LANGUAGE:CXX>:-march=native>")
    endif()
endif()

option(GEC_ENABLE_PTHREADS "use pthread" OFF)
if(GEC_ENABLE_PTHREADS)
    target_compile_definitions(gec INTERFACE GEC_ENABLE_PTHREADS)
    if(WIN32)
        find_package(pthreads CONFIG REQUIRED)
        target_link_libraries(
            gec
            INTERFACE PThreads4W::PThreads4W PThreads4W::PThreads4W_CXXEXC
                      PThreads4W::PThreads4W_STRUCTEXC
                      PThreads_windows::PThreads_windows)
    else()
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(gec INTERFACE Threads::Threads)
    endif()
endif()

if(GEC_ENABLE_CUDA)
    find_package(CUDAToolkit)
    target_link_libraries(gec INTERFACE CUDA::curand)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
