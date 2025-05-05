## Compilation Instructions

The project was created and tested on Linux 6.8.0-57-generic Ubuntu 22.04 with gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 and CMake version 3.22.1.

# Requirements

- Cmake 3.10 or higher
- SFML 2.5 or higher

# Compiling on Linux/MacOS

```bash
mkdir build
cd build
cmake ..
make
```

# Running the program

```bash
./NeuralNetworkMNIST
```

# Compiling on Windows with Visual Studio

```bash
# Create build directory
mkdir build
cd build

# Generate Visual Studio project files
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build using Visual Studio or from command line
cmake --build . --config Release

# Run the application
Release\NeuralNetworkMNIST.exe
```

# Compiling on Windows with MinGW

```bash
# Create build directory
mkdir build
cd build

# Generate build files
cmake -G "MinGW Makefiles" ..

# Build the project
mingw32-make

# Run the application
NeuralNetworkMNIST.exe
```
