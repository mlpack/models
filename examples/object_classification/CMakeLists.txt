cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(object_classification)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  object_classification.cpp
)

if(DEBUG)
  message("Compilation with debug info (with ggdb3 flag)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3")
else()
   message("Compilation without debug info (without ggdb3 flag)")
endif()

foreach(file ${SOURCES})
  string( REPLACE ".cpp" "" name ${file})
  add_executable(${name} ${MODEL_SOURCE_DIR}/${file})
  target_link_libraries(${name}
      ${MLPACK_LIBRARIES}
     ${ARMADILLO_LIBRARIES}
      ${Boost_LIBRARIES})
endforeach()