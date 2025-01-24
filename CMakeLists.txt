cmake_minimum_required(VERSION 3.15)
project(JumpKingAI)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetwork
        src/main.cpp
        src/neural_network.cpp
        src/game_simulation.cpp
        src/neural_network.h
        src/game_simulation.h
)

#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(JumpKingAI ${OpenCV_LIBS})
