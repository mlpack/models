cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(object_classification)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  object_classification.cpp
)

foreach(file ${SOURCES})
  string( REPLACE ".cpp" "" name ${file})
  add_executable(${name} ${MODEL_SOURCE_DIR}/${file})
  target_link_libraries(${name}
      ${COMPILER_SUPPORT_LIBRARIES}
      ${ARMADILLO_LIBRARIES}
      ${Boost_FILESYSTEM_LIBRARY}
      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
      ${Boost_SYSTEM_LIBRARY}
      ${Boost_SERIALIZATION_LIBRARY}
      ${MLPACK_LIBRARIES}
)
endforeach()