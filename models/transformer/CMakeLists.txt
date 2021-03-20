cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(transformer)

set(DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(SOURCES
  decoder.hpp
  decoder_impl.hpp
  encoder.hpp
  encoder_impl.hpp
  transformer.hpp
  transformer_impl.hpp
)

foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)
