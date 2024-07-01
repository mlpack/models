cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(yolov3_tiny_v1)

set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  yolov3_tiny.hpp
  yolov3_tiny_impl.cpp
)
set(EXEC
  main.cpp
)

foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)

