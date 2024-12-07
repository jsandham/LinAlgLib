# LinAlgLib
Linear algebra library

# Build library, examples and tests from source
cd LinAlgLib
mkdir build
cd build
cmake ../
cmake --build .

# Build documentation
cd LinAlgLib
doxygen docs/Doxyfile

# Clang formatting
clang-format.exe -i --style microsoft library/src/LinearSolvers/AMG/*.cpp
