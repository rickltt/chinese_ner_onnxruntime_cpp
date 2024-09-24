cmake_minimum_required(VERSION 3.18)

project(TestNER)

set(ONNXRUNTIME_ROOTDIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.19.0)
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

find_package(Boost REQUIRED)
find_package(glog REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)

FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} onnxruntime glog::glog)
