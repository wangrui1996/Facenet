cmake_minimum_required(VERSION 3.1)
project(main)
#add_definitions(-DCPU_ONLY)

set(CMAKE_CXX_STANDARD 14)
#set(ProjectPath ${PROJECT_SOURCE_DIR})
set(OpenCV_DIR "/home/rui/libs/opencv-3.4/opencv-3.4/build")
set(CAFFE_SSD /home/rui/work/caffe-oil-upload-ssd)
#SET(EXECUTABLE_OUTPUT_PATH ${ProjectPath})
message(STATUS project\ path\ is:\ ${ProjectPath})

find_package(Boost REQUIRED COMPONENTS system filesystem thread)
find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

find_package(CUDA 5.5 QUIET)
message(STATUS ${CUDA_INCLUDE_DIRS})


message(STATUS  ${CUDA_CUDART_LIBRARY}  )

find_package( Protobuf REQUIRED )

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    message(STATUS find.boost)
    message(STATUS ${Boost_INCLUDE_DIRS})
endif()

find_package(BLAS REQUIRED)

message(${CAFFE_SSD}/include)

include_directories(SYSTEM
        ${PROJECT_SOURCE_DIR}/include
        /usr/include/freetype2/
        ${BLAS_INCLUDE_DIR}
        ${CAFFE_SSD}/include
        ${CAFFE_SSD}/build/include
        /usr/local/cuda/include
        /usr/lib/x86_64-linux-gnu
        ${Boost_INCLUDE_DIRS}
)


include(src/CMakeLists.txt)

link_directories(${CAFFE_SSD}/build/lib)
#link_directories(${ProjectPath}/lib)
#add_library(text SHARED CvxText.cpp)
add_executable(Demo main.cpp Mtcnn.cpp Mtcnn.hpp)
target_link_libraries(Demo glog gflags ${Boost_LIBRARIES} caffe ${OpenCV_LIBS} ${BLAS_LIBRARIES} freetype  ${LINKER_LIBS})
message(STATUS "45gg55")
message(STATUS ${LINKER_LIBS})
#target_link_libraries(Firedetct glog gflags ${Boost_LIBRARIES} caffe ${OpenCV_LIBS} freetype blas ${BLAS_LIBRARIES} m)
