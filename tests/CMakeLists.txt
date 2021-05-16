cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(models_test)

include(CTest)
enable_testing()

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")

add_executable(models_test
               main.cpp
               augmentation_tests.cpp
               ffn_model_tests.cpp
               dataloader_tests.cpp
               preprocessor_tests.cpp
               utils_tests.cpp
)

# Link dependencies of test executable.
target_link_libraries(models_test
  ${COMPILER_SUPPORT_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${MLPACK_LIBRARIES}
)

# Add tests to the testing framework.
# Get the list of sources from the test target.
get_target_property(test_sources models_test SOURCES)

# Go through the list of test sources and parse the test suite name.
foreach(test_file ${test_sources})
  # Regex for parsing files with AUTO_TEST_SUITE.
  file(STRINGS ${test_file} test_suite REGEX "TEST_CASE\\(.*")
  if(NOT "${test_suite}" STREQUAL "")
    # Get the substring of test_suite within brackets in test_name.
    string(REGEX MATCH "\\(.*\\)" test_name ${test_suite})
    # Get the substring excluding the brackets, by calculating the indices.
    string(LENGTH ${test_name} end_idx)
    math(EXPR end_idx "${end_idx} - 2")
    string(SUBSTRING ${test_name} "1" ${end_idx} test)
    # Add the test to the testing tool, test is the name of the test suite.
    add_test(NAME ${test} COMMAND models_test -t ${test} WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR})
  endif()
endforeach()  

# So the dll is placed in the same dir as the tests.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_test(NAME "test" COMMAND models_test)
