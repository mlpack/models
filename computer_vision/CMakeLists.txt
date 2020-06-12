cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(computer_vision)

add_subdirectory(object_classification/)

# Add directory name to sources.
set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)

foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)
