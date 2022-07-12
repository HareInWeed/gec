cmake_minimum_required(VERSION 3.17)

add_executable(unit_test)
target_include_directories(unit_test PRIVATE "../include")
aux_source_directory("." unit_test_files)
target_sources(unit_test PRIVATE ${unit_test_files})
target_link_libraries(unit_test PRIVATE unit_test_shared)
add_test(NAME gec_test COMMAND unit_test ~[bench])
