file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)


add_library(mtcnn ${srcs})
list(APPEND LINKER_LIBS mtcnn)