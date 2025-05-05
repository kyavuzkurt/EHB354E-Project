cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkMNIST VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find SFML package
find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)

# Create source files variable
set(SOURCES
    src/main.cpp
    src/Neuron.cpp
    src/Layer.cpp
    src/Network.cpp
    src/Input.cpp
    src/Button.cpp
    src/NetworkVisualizer.cpp
    # Add other source files as needed
)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE include)

# Link SFML libraries
target_link_libraries(${PROJECT_NAME} PRIVATE sfml-system sfml-window sfml-graphics)

# Copy resources to build directory
file(COPY ${CMAKE_SOURCE_DIR}/resources DESTINATION ${CMAKE_BINARY_DIR})

# Set compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
endif()

# Enable optimization for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Print status message
message(STATUS "Project ${PROJECT_NAME} configured")
