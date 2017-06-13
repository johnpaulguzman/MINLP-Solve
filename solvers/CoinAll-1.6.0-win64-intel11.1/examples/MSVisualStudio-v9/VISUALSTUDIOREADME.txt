The solution file in this directory and the project files associated with the individual examples
are intended to be used with Microsoft Visual Studio version 9 (MSVS 2008) only. They require the
library files and headers that come with the binary distribution, and the executables they build
make certain further assumptions about the directory structure.

It is advisable _not_ to move the executables from the location where the Visual Studio build
puts them, which is in the source directory associated with each example.

For instance, the executable algorithmicDiff.exe will be found in the folder
examples\OSAlgorithmicDiffTest. 

The Visual Studio project files use relative paths to the header files and libraries.
If you want to make change the directory structure of the examples folder consider
setting the INCLUDE and LIB environment variables to point to the proper directories.