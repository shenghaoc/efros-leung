# Non-parametric Texture Synthesis
Coursework for EE4212 Computer Vision. Unlike most other code I've written, this program actually offers superior performance to a standard implementation. OpenCV has to be installed to compile the code but it reduces runtime from hours to minutes for large window sizes.

I got the idea from [UMD CMSC733 PS2](http://www.cs.umd.edu/~djacobs/CMSC733/PS2.pdf) and, according to p. 21 of Efros's [thesis](https://people.eecs.berkeley.edu/~efros/research/efros-thesis.pdf), originally the most time-consuming step of the algorithm was done with convolutions rather than the straightforward quadruple loop suggested by the pseudocode on his [website](http://people.eecs.berkeley.edu/~efros/research/NPS/alg.html).

My understanding of the underlying maths is dismal, but OpenCV implements template matching, and by supplying a flipped template the output of the function matches that of the quadruple loop.

I believe the increase in speed comes from the fast Fourier transform (FFT), and BLAS routines which optimize functions for linear algebra in hardware.

**Running the code**

After compiling the program, run it with an odd number as the argument for the size of the neighbourhood window.

The program will search for JPEG files beginning with the prefix "texture". After synthesis, the corresponding output file will have the same name but with the prefix changed to "synth".

**Setting up OpenCV**

~~For [Linux](https://docs.opencv.org/4.5.1/d7/d9f/tutorial_linux_install.html) and [macOS](https://docs.opencv.org/4.5.1/d0/db2/tutorial_macos_install.html), I have included a [CMakeLists.txt](EE4212Part2Assignment1/CMakeLists.txt) file. On macOS, `OpenCV_DIR` is set manually in this file because the output of OpenCV compilation is not installed. I compiled mine in `~/opt/build_opencv`. Compilation of this OpenCV program is the same as that of a [typical CMake project](https://docs.opencv.org/4.5.1/db/df5/tutorial_linux_gcc_cmake.html).~~

~~For [Windows](https://docs.opencv.org/4.5.1/d3/d52/tutorial_windows_install.html), I have included two property sheets [OpenCV_Debug.props](EE4212Part2Assignment1/OpenCV_Debug.props) and [OpenCV_Release.props](EE4212Part2Assignment1/OpenCV_Release.props). Debugging makes the program significantly slower but appears to provide helpful checks and clearer error messages. I referred to [this guide](https://mathcs.clarku.edu/~jmagee/cs262/examples/OpenCV-with-Visual-Studio2017.pdf) as the [official one](https://docs.opencv.org/4.5.1/dd/d6e/tutorial_windows_visual_studio_opencv.html) appeared out-of-date. However, I did create property sheets for future use. Note that for *Additional Dependencies* the name of the library file includes the specific OpenCV version number. I used OpenCV 4.5.1.~~

Switched to vcpkg, should be able to download and compile automatically with IDEs that support CMake. Only issue is that instead of precompiled binaries everything is compiled locally.