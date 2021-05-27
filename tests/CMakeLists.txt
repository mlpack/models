cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(models_test)

include(CTest)
enable_testing()
include(${CMAKE_SOURCE_DIR}/CMake/Catch.cmake)

# The Catch CMake tools do not seem to work correctly on windows.
if(NOT WIN32)
  include(${CMAKE_SOURCE_DIR}/CMake/Catch.cmake)
endif()

# The Catch CMake tools do not seem to work correctly on windows.
if(NOT WIN32)
  include(${CMAKE_SOURCE_DIR}/CMake/Catch.cmake)
endif()

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

# So the dll is placed in the same dir as the tests.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT WIN32)
  catch_discover_tests(models_test)
else()
  add_test(NAME "test" COMMAND models_test)
endif()
