cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(models_test)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")

add_executable(models_test
               utils_tests.cpp
               dataloader_tests.cpp
)

# Link dependencies of test executable.
target_link_libraries(models_test
  ${MLPACK_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${COMPILER_SUPPORT_LIBRARIES}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${CURL_LIBRARIES}
)

# Add tests to the testing framework
# Get the list of sources from the test target
get_target_property(test_sources models_test SOURCES)

# Go through the list of test sources and parse the test suite name
foreach(test_file ${test_sources})
  # Regex for parsing files with AUTO_TEST_SUITE
  file(STRINGS ${test_file} test_suite REGEX "BOOST_AUTO_TEST_SUITE\\(.*")
  if(NOT "${test_suite}" STREQUAL "")
    # Get the substring of test_suite within brackets in test_name
    string(REGEX MATCH "\\(.*\\)" test_name ${test_suite})
    # Get the substring excluding the brackets, by calculating the indices
    string(LENGTH ${test_name} end_idx)
    math(EXPR end_idx "${end_idx} - 2")
    string(SUBSTRING ${test_name} "1" ${end_idx} test)
    # Add the test to the testing tool, test is the name of the test suite
    add_test(NAME ${test} COMMAND models_test -t ${test} WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR})
  endif()
  # Regex for parsing files with FIXTURE_TEST_SUITE similarly
  file(STRINGS ${test_file} test_suite REGEX "BOOST_FIXTURE_TEST_SUITE\\(.*,")
  if(NOT "${test_suite}" STREQUAL "")
    # Get the substring of test_suite within brackets and comma in test_name
    string(REGEX MATCH "\\(.*," test_name ${test_suite})
    string(LENGTH ${test_name} end_idx)
    math(EXPR end_idx "${end_idx} - 2")
    string(SUBSTRING ${test_name} "1" ${end_idx} test)
    add_test(NAME ${test} COMMAND models_test -t ${test} WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR})
  endif()
endforeach()

set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)