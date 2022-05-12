cmake_minimum_required(VERSION 3.15)

find_package(Catch2 CONFIG REQUIRED)

add_executable(unit_test)

target_sources(
    unit_test
    PRIVATE "test_main.cpp"
            "common.cpp"
            "curve.cpp"
            "test_arithmetic.cpp"
            "test_bigint.cpp"
            "test_field.cpp"
            "test_context.cpp"
            "test_curve.cpp")

target_compile_features(unit_test PRIVATE cxx_std_14)
target_link_libraries(unit_test PRIVATE gec)
target_link_libraries(unit_test PRIVATE Catch2::Catch2)

add_test(NAME gec_test COMMAND unit_test ~[bench])

if(GEC_ENABLE_CUDA)
    add_executable(cu_unit_test)

    target_sources(cu_unit_test PRIVATE "test_main.cpp" "test_common.cpp"
                                        "test_field.cu")

    target_compile_features(cu_unit_test PRIVATE cxx_std_14)
    target_link_libraries(cu_unit_test PRIVATE gec)
    target_link_libraries(cu_unit_test PRIVATE Catch2::Catch2)
    set_target_properties(cu_unit_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_test(NAME gec_cuda_test COMMAND cu_unit_test ~[bench])
endif()
