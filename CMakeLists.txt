cmake_minimum_required(VERSION 3.7)
project(face_rcg)
FIND_PACKAGE(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/face_features/detect_face.cpp
        src/face_features/detect_face.h
        src/face_features/load_data.cpp
        src/face_features/load_data.h
        src/face_features/pre_treat.cpp
        src/face_features/pre_treat.h
        src/face_features/test.cpp
        src/face_features/test.h
        src/main.cpp)

add_executable(face_rcg ${SOURCE_FILES})

target_link_libraries(face_rcg ${OpenCV_LIBS})