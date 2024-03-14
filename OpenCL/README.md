# Topics in Parallel Computing - Assignment 2 (OpenCL)

## Description

For the class' second assignment of the semester, we were again given a program consisting of the serial implementation of an algorithm which performs Gaussian blur (gaussian-blur.c, just like in the [first assignment](https://github.com/hvlkk/AUEB-Projects/tree/main/Topics-In-Parallel-Programming/Assignment-1)), which we had to parallelise using [OpenCL](https://www.khronos.org/opencl/), and compare our findings against the serial algorithm and the OpenMP algorithm we had coded for the previous assignment. Through OpenCL, the computations were shifted to the GPU for accelerated processing.

### gaussian-blur.c

Just like the previous assignment, the algorithm applies Gaussian blur to a BMP image file. Again, the image dimensions must be square (equal width and height) and the image should be named "size.bmp" (where size represents the width/height of the image) for the algorithm to function correctly. The program is run with 2 command line arguments: `blur_radius image_size`, where blur_radius represents the radius used during the application of Gaussian blur, and image_size represents the image size as specified in the previous sentence (without the ".bmp" file extension). The program is again run with 2 command line arguments: blur_radius image_size, where blur_radius represents the radius used during the application of Gaussian blur, and image_size represents the image size as specified in the previous sentence (without the ".bmp" file extension).

Here, OpenCL was used to parallelise the algorithm, with the computations happening in the GPU instead of the CPU. A char pointer was used to write the necessary kernel source code (to be executed in the GPU), instead of a separate .cl file. Afterwards, OpenCL memory objects are created, the kernel code is compiled with the appropriate arguments, and the code is executed on the GPU.

## System Requirements

To run this project, a gcc compiler is again required, and OpenMP needs to be enabled in the settings of the IDE you choose to run the project on. The version of gcc that was used throughout the development of the project and the project has been confirmed to work on is 8.1.0. In addition, OpenCL needs to have been installed, and the necessary libraries need to have been linked to the project. Refer to the [OpenCL website](https://www.khronos.org/opencl/) for more information.

As a further warning, the program (gaussian-blur.c) is configured to use 4 threads. Therefore, if the CPU of the machine you are using to run this project has fewer than 4 cores, you are advised to replace `num_threads(4)` with `num_threads(x)` before you run each program, where x represents the amount of cores your CPU has.

## License

[MIT](https://choosealicense.com/licenses/mit/)
