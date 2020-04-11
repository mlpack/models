cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(object_classification)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  object_classification.cpp
  loadmodel.cpp
)

foreach(file ${SOURCES})
  string( REPLACE ".cpp" "" name ${file})
  add_executable(${name} ${MODEL_SOURCE_DIR}/${file})
  target_link_libraries(${name}
      ${MLPACK_LIBRARIES}
      ${ARMADILLO_LIBRARIES}
      ${Boost_LIBRARIES}
      ${CURL_LIBRARIES})
endforeach()