cmake_minimum_required(VERSION 3.17)

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

add_library(gec INTERFACE)
target_include_directories(gec INTERFACE "./include")
target_compile_features(gec INTERFACE cxx_std_14)

add_library(secp256k1 STATIC "src/secp256k1.cpp")
target_link_libraries(secp256k1 PUBLIC gec)

add_library(sm2 STATIC "src/sm2.cpp")
target_link_libraries(sm2 PUBLIC gec)

add_library(curves INTERFACE)
target_link_libraries(curves INTERFACE secp256k1 sm2)

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
    target_compile_definitions(gec INTERFACE GEC_ENABLE_CUDA)

    set_source_files_properties("src/secp256k1.cpp" PROPERTIES LANGUAGE CUDA)
    set_target_properties(secp256k1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(secp256k1 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_source_files_properties("src/sm2.cpp" PROPERTIES LANGUAGE CUDA)
    set_target_properties(sm2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(sm2 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Only do these if this is the main project, and not if it is included through
# `add_subdirectory`
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)

    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()

    if(GEC_ENABLE_BENCH)
        add_subdirectory(bench)
    endif()
endif()
