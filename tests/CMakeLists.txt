cmake_minimum_required(VERSION 3.17)

find_package(Catch2 CONFIG REQUIRED)

add_library(unit_test_shared STATIC "test_main.cpp")
target_compile_features(unit_test_shared PUBLIC cxx_std_14)
target_link_libraries(unit_test_shared PUBLIC gec Catch2::Catch2)

# enable warning in test
target_compile_options(
    unit_test_shared
    PUBLIC
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall
    -Wextra
    -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Xcompiler=-Wall
    -Xcompiler=-Wextra>
)

add_subdirectory(cpu)

if(GEC_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
