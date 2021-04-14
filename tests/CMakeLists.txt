cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(models_test)

enable_testing()

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")

add_executable(models_test
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
  ${Boost_SERIALIZATION_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${MLPACK_LIBRARIES}
)

# Add tests to the testing framework.
# Get the list of sources from the test target.
get_target_property(test_sources models_test SOURCES)

#set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)

add_test(NAME "test" COMMAND models_test)