cmake_minimum_required(VERSION 3.17)

find_package(Catch2 CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

add_library(bench_shared STATIC "bench_main.cpp")
target_compile_features(bench_shared PUBLIC cxx_std_14)
target_link_libraries(bench_shared PUBLIC Catch2::Catch2)

add_executable(bench_gec_cpu "bench_gec_cpu.cpp")
target_include_directories(bench_gec_cpu PRIVATE "./include")
target_link_libraries(bench_gec_cpu PRIVATE gec curves bench_shared)

add_executable(bench_openssl "bench_openssl.cpp")
target_include_directories(bench_openssl PRIVATE "./include")
target_link_libraries(bench_openssl PRIVATE OpenSSL::SSL OpenSSL::Crypto bench_shared)

if(GEC_ENABLE_CUDA)
    add_executable(bench_gec_cuda "bench_gec_cuda.cu")
    target_include_directories(bench_gec_cuda PRIVATE "./include")
    target_link_libraries(bench_gec_cuda PRIVATE gec curves bench_shared)
endif()
