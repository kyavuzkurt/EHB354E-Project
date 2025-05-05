## Compilation Instructions

The project was created and tested on Linux 6.8.0-57-generic Ubuntu 22.04 with g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0, SFML 2.5.1 and CMake version 3.22.1.

# Requirements

- CMake 3.10 or higher
- SFML 2.5 or higher
- Data files in the `data` directory:
  - `mnist_data_train.csv`
  - `mnist_data_test.csv`
- Resources in the `resources` directory:
  - `font.ttf`

# Compiling on Linux/MacOS

```bash
# Install SFML (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libsfml-dev

# Create build directory
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
# Download SFML
# Get SFML 2.5.1 from https://www.sfml-dev.org/files/SFML-2.5.1-windows-vc15-64-bit.zip
# Extract to C:/SFML/SFML-2.5.1 or your preferred location

# Create build directory
mkdir build
cd build

# Generate Visual Studio project files (adjust Visual Studio version as needed)
cmake -G "Visual Studio 17 2022" -A x64 -DSFML_DIR=C:/SFML/SFML-2.5.1/lib/cmake/SFML ..

# Build using Visual Studio or from command line
cmake --build . --config Release

# Copy SFML DLLs to executable location
# Copy all DLLs from C:/SFML/SFML-2.5.1/bin/ to your Release directory

# Run the application
Release\NeuralNetworkMNIST.exe
```

# Automated Builds

This project uses GitHub Actions for automated builds on Ubuntu and Windows. The workflow:
- Installs SFML on each platform
- Creates necessary data and resource directories
- Builds the project with CMake
- Verifies the build output

For more details, see the workflow configuration in `.github/workflows/cmake-multi-platform.yml`.
