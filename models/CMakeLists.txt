cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(models)

set(DIR
  darknet
  yolo
  resnet)

foreach(dir ${DIR})
  add_subdirectory(${dir})
endforeach()

# Add directory name to sources.
set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)

foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/layer.hpp)
# Append sources (with directory name) to list of all models sources (used at
# the parent scope).
set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)
