cmake_minimum_required(VERSION 3.15)

add_executable(unit_test)

target_sources(unit_test
    PRIVATE "test_main.cpp"
            "test_common.cpp"
            "test_arithmetic.cpp"
            "test_bigint.cpp"
            "test_field.cpp")

target_compile_features(unit_test PRIVATE cxx_std_14)

if(GEC_ENABLE_CUDA)
    set_target_properties(unit_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(unit_test PRIVATE gec)

find_package(Catch2 CONFIG REQUIRED)
target_link_libraries(unit_test PRIVATE Catch2::Catch2)

add_test(NAME gec_test COMMAND unit_test)
