cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(dataloaders)

option(DEBUG "DEBUG" OFF)

set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
    preprocessor.hpp
    datasets.hpp
    dataloader.hpp
    dataloader_impl.hpp
)

foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# Append sources (with directory name) to list of all models sources (used at
# the parent scope).
set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)
