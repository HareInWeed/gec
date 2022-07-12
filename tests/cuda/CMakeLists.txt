cmake_minimum_required(VERSION 3.17)

add_executable(cu_unit_test)
target_include_directories(cu_unit_test PRIVATE "../cpu")
target_include_directories(cu_unit_test PRIVATE "../include")
aux_source_directory("." cu_unit_test_files)
target_sources(cu_unit_test PRIVATE ${cu_unit_test_files})
target_link_libraries(cu_unit_test PRIVATE unit_test_shared)
set_target_properties(cu_unit_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cu_unit_test PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>)
target_compile_options(cu_unit_test PRIVATE
    $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)

# target_compile_options(cu_unit_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number>)
# target_compile_options(cu_unit_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --no_display_error_number>)
add_test(NAME gec_cuda_test COMMAND cu_unit_test ~[bench])
