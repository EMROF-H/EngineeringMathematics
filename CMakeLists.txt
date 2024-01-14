cmake_minimum_required(VERSION 3.26)
project(EngineeringMathematics)

set(CMAKE_CXX_STANDARD 23)

set(BIN ${PROJECT_SOURCE_DIR}/bin)

set(LIBRARY_OUTPUT_PATH ${BIN})
set(EXECUTABLE_OUTPUT_PATH ${BIN})

add_executable(main main.cpp)
set_target_properties(main PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(main PUBLIC include)
