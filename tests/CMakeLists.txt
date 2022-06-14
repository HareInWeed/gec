cmake_minimum_required(VERSION 3.15)

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
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Xcompiler="-Wall -Wextra -Wpedantic">
)

add_executable(unit_test)
target_sources(
    unit_test
    PRIVATE "common.cpp"
            "curve.cpp"
            "test_misc.cpp"
            "test_arithmetic.cpp"
            "test_bigint.cpp"
            "test_field.cpp"
            "test_context.cpp"
            "test_curve.cpp"
            "test_dlp.cpp")
target_link_libraries(unit_test PRIVATE unit_test_shared)

add_test(NAME gec_test COMMAND unit_test ~[bench])

if(GEC_ENABLE_CUDA)
    add_executable(cu_unit_test)
    target_sources(
        cu_unit_test
        PRIVATE "cuda_common.cu"
                "common.cu"
                "curve.cu"
                "test_field.cu"
                "cu_test_field.cu"
                "test_dlp.cu"
                "cu_test_dlp.cu")
    target_link_libraries(cu_unit_test PRIVATE unit_test_shared)
    set_target_properties(cu_unit_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # target_compile_options(cu_unit_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number>)

    add_test(NAME gec_cuda_test COMMAND cu_unit_test ~[bench])
endif()
