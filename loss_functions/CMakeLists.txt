cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(loss_functions)

set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)

set(SOURCES
    yolo_loss.hpp
    yolo_loss_impl.hpp
)

foreach(file ${SOURCES})
   set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# Append sources (with directory name) to list of all models sources (used at
# the parent scope).
set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)
