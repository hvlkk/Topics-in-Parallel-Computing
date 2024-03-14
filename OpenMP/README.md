# Topics in Parallel Computing - Assignment 1 (OpenMP)

## Description

For the class' first assignment of the semester, we were given two programs (primes.c and gaussian-blur.c), both of which only consisted of serial implementations, that we had to parallelise using [OpenMP](https://www.openmp.org/).

### primes.c

For the first part of the assignment, we were given the serial implementation of a program that, given a certain number (defined in the UPTO symbolic constant), calculates the amount of prime numbers between 0 and that number, as well as the last prime number found in the aforementioned range. Then, we were asked to code a function that utilises OpenMP preprocessor directives above the provided serial implementation, in order to parallelise the execution.

### gaussian-blur.c

For the second part of the assignment, we were again given the serial implementation of a program, which this time applies Gaussian blur to a BMP image file. The image dimensions must be square (equal width and height) and the image should be named "size.bmp" (where size represents the width/height of the image) for the algorithm to function correctly. The program is run with 2 command line arguments: `blur_radius image_size`, where blur_radius represents the radius used during the application of Gaussian blur, and image_size represents the image size as specified in the previous sentence (without the ".bmp" file extension).

Here, we were asked to provide 2 implementations in order to parallelise the serial implementations; one through OpenMP loops, and one through OpenMP tasks. Again, we were not allowed to modify the code for the serial algorithm.

## System Requirements

To run this project, a gcc compiler is required, and OpenMP needs to be enabled in the settings of the IDE you choose to run the project on. The version of gcc that was used throughout the development of the project and the project has been confirmed to work on is 8.1.0.

As a further warning, the two programs (primes.c and gaussian-blur.c) are configured to use 4 threads. Therefore, if the CPU of the machine you are using to run this project has fewer than 4 cores, you are advised to replace `num_threads(4)` with `num_threads(x)` before you run each program, where x represents the amount of cores your CPU has.

## License

[MIT](https://choosealicense.com/licenses/mit/)
