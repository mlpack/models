cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(models_test)

set(MODEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")

set(SOURCES
  dataloader_tests.cpp
  utils_tests.cpp
  model_tests.cpp
)

foreach(file ${SOURCES})
  string( REPLACE ".cpp" "" name ${file})
  add_executable(${name} ${MODEL_SOURCE_DIR}/${file})
  target_link_libraries(${name}
      ${MLPACK_LIBRARIES}
      ${ARMADILLO_LIBRARIES}
      ${Boost_LIBRARIES})
endforeach()

foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(DIRS ${DIRS} ${DIR_SRCS} PARENT_SCOPE)