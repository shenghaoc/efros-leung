# Non-parametric Texture Synthesis
Coursework for EE4212 Computer Vision. Unlike most other code I've written, this program actually offers superior performance to a standard implementation. OpenCV has to be installed to compile the code but it reduces runtime from hours to minutes for large window sizes.

I got the idea from [UMD CMSC733 PS2](http://www.cs.umd.edu/~djacobs/CMSC733/PS2.pdf) and, according to p. 21 of Efros's [thesis](https://people.eecs.berkeley.edu/~efros/research/efros-thesis.pdf), originally the most time-consuming step of the algorithm was done with convolutions rather than the straightforward quadruple loop suggested by the pseudocode on his [website](http://people.eecs.berkeley.edu/~efros/research/NPS/alg.html).

My understanding of the underlying maths is dismal, but OpenCV implements template matching, and by supplying a flipped template the output of the function matches that of the quadruple loop.

I believe the increase in speed comes from the fast Fourier transform (FFT), and BLAS routines which optimize functions for linear algebra in hardware.

**Running the code**

After compiling the program, run it with an odd number as the argument for the size of the neighbourhood window.

The program will search for JPEG files beginning with the prefix "texture". After synthesis, the corresponding output file will have the same name but with the prefix changed to "synth".
