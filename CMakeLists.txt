cmake_minimum_required(VERSION 3.20)

# Project Name
project(JumpKingAI)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the OpenCV directory
set(OpenCV_DIR "C:/opencv/build")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories for OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your source files
set(SOURCE_FILES
        src/main.cpp
        src/game_simulation.cpp
        src/neural_network.cpp
)

# Add the executable
add_executable(JumpKingAI ${SOURCE_FILES})

# Link OpenCV libraries
target_link_libraries(JumpKingAI ${OpenCV_LIBS})

# Set the library search path for linking OpenCV (explicitly)
link_directories("C:/opencv/build/x64/vc16/lib")

# Optional: Output directories for build files
set_target_properties(JumpKingAI PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Print helpful information during configuration
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include Directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
