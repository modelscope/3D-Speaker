cmake_minimum_required(VERSION 3.23)
project(SpeakerLabEngines VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)

option(USE_CUDA "Build with CUDA support" OFF)

include_directories(${PROJECT_SOURCE_DIR})

# Fetch third-party library
#include(ExternalProject)
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/third_party)
if (NOT EXISTS ${FETCHCONTENT_BASE_DIR})
    file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR})
endif ()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(cmake/build_json.cmake)
include(cmake/build_onnx.cmake)

add_subdirectory(utils)
add_subdirectory(bin)
add_subdirectory(feature)
add_subdirectory(model)

