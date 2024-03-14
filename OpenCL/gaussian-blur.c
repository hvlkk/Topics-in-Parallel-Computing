#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#pragma pack(push, 2)
	typedef struct bmpheader_
	{
		char sign;
		int size;
		int notused;
		int data;
		int headwidth;
		int width;
		int height;
		short numofplanes;
		short bitpix;
		int method;
		int arraywidth;
		int horizresol;
		int vertresol;
		int colnum;
		int basecolnum;
	} bmpheader_t;
#pragma pack(pop)

const char* kernelSource =
"__kernel void opencl_gaussian_blur("
"__global const unsigned char* imgin_red, __global const unsigned char* imgin_green, __global const unsigned char* imgin_blue,"
"__global unsigned char* imgout_red, __global unsigned char* imgout_green, __global unsigned char* imgout_blue, int radius)"

"{"
"	int i = get_global_id(0);"
"	int j = get_global_id(1);"
"	int width =  get_global_size(0);"
"	int height = get_global_size(1);"
"	double row, col;"
"	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;"
"	for (row = i - radius; row <= i + radius; row++) {"
"		for (col = j - radius; col <= j + radius; col++) {"
"			int x = clamp((int) col, 0, width - 1);"
"			int y = clamp((int) row, 0, height - 1);"
"			int tempPos = y * width + x;"
"			double square = (col - j) * (col - j) + (row - i) * (row - i);"
"			double sigma = radius * radius;"
"			double weight = exp(-square / (2 * sigma)) / (3.14 * 2 * sigma);"

"			redSum += imgin_red[tempPos] * weight;"
"			greenSum += imgin_green[tempPos] * weight;"
"			blueSum += imgin_blue[tempPos] * weight;"
"			weightSum += weight;"
"		}"
"	}"

"	int pos = i * width + j;"
"	imgout_red[pos] = round(redSum / weightSum);"
"	imgout_green[pos] = round(greenSum / weightSum);"
"	imgout_blue[pos] = round(blueSum / weightSum);"
"}";

/* This is the image structure, containing all the BMP information
 * plus the RGB channels.
 */
typedef struct img_
{
	bmpheader_t header;
	int rgb_width;
	unsigned char *imgdata;
	unsigned char *red;
	unsigned char *green;
	unsigned char *blue;
} img_t;

void gaussian_blur_serial(int, img_t *, img_t *);
void gaussian_blur_omp(int, img_t *, img_t *);
void gaussian_blur_opencl(int, img_t *, img_t *);


/* START of BMP utility functions */
static
void bmp_read_img_from_file(char *inputfile, img_t *img)
{
	FILE *file;
	bmpheader_t *header = &(img->header);

	file = fopen(inputfile, "rb");
	if (file == NULL)
	{
		fprintf(stderr, "File %s not found; exiting.", inputfile);
		exit(1);
	}

	fread(header, sizeof(bmpheader_t)+1, 1, file);
	if (header->bitpix != 24)
	{
		fprintf(stderr, "File %s is not in 24-bit format; exiting.", inputfile);
		exit(1);
	}

	img->imgdata = (unsigned char*) calloc(header->arraywidth, sizeof(unsigned char));
	if (img->imgdata == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for image data; exiting.");
		exit(1);
	}

	fseek(file, header->data, SEEK_SET);
	fread(img->imgdata, header->arraywidth, 1, file);
	fclose(file);
}

static
void bmp_clone_empty_img(img_t *imgin, img_t *imgout)
{
	imgout->header = imgin->header;
	imgout->imgdata =
		(unsigned char*) calloc(imgout->header.arraywidth, sizeof(unsigned char));
	if (imgout->imgdata == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for clone image data; exiting.");
		exit(1);
	}
}

static
void bmp_write_data_to_file(char *fname, img_t *img)
{
	FILE *file;
	bmpheader_t *bmph = &(img->header);

	file = fopen(fname, "wb");
	fwrite(bmph, sizeof(bmpheader_t)+1, 1, file);
	fseek(file, bmph->data, SEEK_SET);
	fwrite(img->imgdata, bmph->arraywidth, 1, file);
	fclose(file);
}

static
void bmp_rgb_from_data(img_t *img)
{
	bmpheader_t *bmph = &(img->header);

	int i, j, pos = 0;
	int width = bmph->width, height = bmph->height;
	int rgb_width = img->rgb_width;

	for (i = 0; i < height; i++)
		for (j = 0; j < width * 3; j += 3, pos++)
		{
			img->red[pos]   = img->imgdata[i * rgb_width + j];
			img->green[pos] = img->imgdata[i * rgb_width + j + 1];
			img->blue[pos]  = img->imgdata[i * rgb_width + j + 2];
		}
}

static
void bmp_data_from_rgb(img_t *img)
{
	bmpheader_t *bmph = &(img->header);
	int i, j, pos = 0;
	int width = bmph->width, height = bmph->height;
	int rgb_width = img->rgb_width;

	for (i = 0; i < height; i++ )
		for (j = 0; j < width* 3 ; j += 3 , pos++)
		{
			img->imgdata[i * rgb_width  + j]     = img->red[pos];
			img->imgdata[i * rgb_width  + j + 1] = img->green[pos];
			img->imgdata[i * rgb_width  + j + 2] = img->blue[pos];
		}
}

static
void bmp_rgb_alloc(img_t *img)
{
	int width, height;

	width = img->header.width;
	height = img->header.height;

	img->red = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->red == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the red channel; exiting.");
		exit(1);
	}

	img->green = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->green == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the green channel; exiting.");
		exit(1);
	}

	img->blue = (unsigned char*) calloc(width*height, sizeof(unsigned char));
	if (img->blue == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for the blue channel; exiting.");
		exit(1);
	}

	img->rgb_width = width * 3;
	if ((width * 3  % 4) != 0) {
	   img->rgb_width += (4 - (width * 3 % 4));
	}
}

static
void bmp_img_free(img_t *img)
{
	free(img->red);
	free(img->green);
	free(img->blue);
	free(img->imgdata);
}

/* END of BMP utility functions */

/* check bounds */
int clamp(int i , int min , int max)
{
	if (i < min) return min;
	else if (i > max) return max;
	return i;
}


/* Sequential Gaussian Blur */
void gaussian_blur_serial(int radius, img_t *imgin, img_t *imgout)
{
	int i, j;
	int width = imgin->header.width, height = imgin->header.height;
	double row, col;
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width ; j++)
		{
			for (row = i-radius; row <= i + radius; row++)
			{
				for (col = j-radius; col <= j + radius; col++)
				{
					int x = clamp(col, 0, width-1);
					int y = clamp(row, 0, height-1);
					int tempPos = y * width + x;
					double square = (col-j)*(col-j)+(row-i)*(row-i);
					double sigma = radius*radius;
					double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

					redSum += imgin->red[tempPos] * weight;
					greenSum += imgin->green[tempPos] * weight;
					blueSum += imgin->blue[tempPos] * weight;
					weightSum += weight;
				}
			}
			imgout->red[i*width+j] = round(redSum/weightSum);
			imgout->green[i*width+j] = round(greenSum/weightSum);
			imgout->blue[i*width+j] = round(blueSum/weightSum);

			redSum = 0;
			greenSum = 0;
			blueSum = 0;
			weightSum = 0;
		}
	}
}


/* Parallel Gaussian Blur with OpenMP loop parallelization */
void gaussian_blur_omp_loops(int radius, img_t *imgin, img_t *imgout)
{
	/* TODO: Implement parallel Gaussian Blur using OpenMP loop parallelization */
	omp_set_dynamic(0);     // again, to ensure we have the exact amount of threads intended

    int i, j;
	int width = imgin->header.width, height = imgin->header.height;
	double row, col;
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

	#pragma omp parallel for num_threads(4) schedule(runtime) private(row, col, i, j) firstprivate(weightSum, redSum, greenSum, blueSum)
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width ; j++)
		{
			for (row = i-radius; row <= i + radius; row++)
			{
				for (col = j-radius; col <= j + radius; col++)
				{
					int x = clamp(col, 0, width-1);
					int y = clamp(row, 0, height-1);
					int tempPos = y * width + x;
					double square = (col-j)*(col-j)+(row-i)*(row-i);
					double sigma = radius*radius;
					double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

					redSum += imgin->red[tempPos] * weight;
					greenSum += imgin->green[tempPos] * weight;
					blueSum += imgin->blue[tempPos] * weight;
					weightSum += weight;
				}
			}
			imgout->red[i*width+j] = round(redSum/weightSum);
			imgout->green[i*width+j] = round(greenSum/weightSum);
			imgout->blue[i*width+j] = round(blueSum/weightSum);

			redSum = 0;
			greenSum = 0;
			blueSum = 0;
			weightSum = 0;
		}
	}
}


/* Parallel Gaussian Blur with OpenMP tasks */
void gaussian_blur_omp_tasks(int radius, img_t *imgin, img_t *imgout)
{
	/* TODO: Implement parallel Gaussian Blur using OpenMP tasks */
	int i, j;
	int width = imgin->header.width, height = imgin->header.height;
	double row, col;
	double weightSum = 0.0, redSum = 0.0, greenSum = 0.0, blueSum = 0.0;

	omp_set_dynamic(0);     // yet again, to ensure we have the exact amount of threads intended

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (i = 0; i < height; i++)
            {
                # pragma omp task firstprivate(i) private(j, row, col, weightSum, redSum, greenSum, blueSum)
                for (j = 0; j < width ; j++)
                {
                    for (row = i-radius; row <= i + radius; row++)
                    {
                        for (col = j-radius; col <= j + radius; col++)
                        {
                            int x = clamp(col, 0, width-1);
                            int y = clamp(row, 0, height-1);
                            int tempPos = y * width + x;
                            double square = (col-j)*(col-j)+(row-i)*(row-i);
                            double sigma = radius*radius;
                            double weight = exp(-square / (2*sigma)) / (3.14*2*sigma);

                            redSum += imgin->red[tempPos] * weight;
                            greenSum += imgin->green[tempPos] * weight;
                            blueSum += imgin->blue[tempPos] * weight;
                            weightSum += weight;
                        }
                    }
                    imgout->red[i*width+j] = round(redSum/weightSum);
                    imgout->green[i*width+j] = round(greenSum/weightSum);
                    imgout->blue[i*width+j] = round(blueSum/weightSum);

                    redSum = 0;
                    greenSum = 0;
                    blueSum = 0;
                    weightSum = 0;
                }
            }
        }
    }
}

/* Parallel Gaussian Blur with OpenCL */

// getErrorString: Function used while debugging. Returns OpenCL errors verbatim.
const char* getErrorString(cl_int err)
{
	switch (err) {
	// run-time and JIT compiler errors
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

	// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

	// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

/* Parallel Gaussian Blur with OpenCL */
void gaussian_blur_opencl(int radius, img_t* imgin, img_t* imgout)
{
	// OpenCL setup
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue commandQueue;
	cl_program program;
	cl_kernel kernel;
	cl_mem imginRedBuf, imginGreenBuf, imginBlueBuf, imgoutRedBuf, imgoutGreenBuf, imgoutBlueBuf;

	cl_int err;

	// Create the OpenCL context
	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

	// Create the command queue
	commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	// Create the program from the kernel source code
	program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
	clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	// Create the kernel
	kernel = clCreateKernel(program, "opencl_gaussian_blur", &err);

	// Create the buffers:
	// Fetching the width & height of the image, to allocate memory appropriately
	int width = imgin->header.width;
	int height = imgin->header.height;
	size_t pixels = width * height * sizeof(unsigned char);

    // imgin buffers will also need to copy the host pointers
	imginRedBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * pixels, imgin->red, &err);
	imginGreenBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * pixels, imgin->green, &err);
	imginBlueBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * pixels, imgin->blue, &err);

	// imgout buffers will not need to be copied from host, since we will be writing custom results to the entire stretch of the allocated memory
	imgoutRedBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * pixels, NULL, &err);
	imgoutGreenBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * pixels, NULL, &err);
	imgoutBlueBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * pixels, NULL, &err);

	// Set the kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&imginRedBuf);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&imginGreenBuf);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&imginBlueBuf);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&imgoutRedBuf);
	err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&imgoutGreenBuf);
	err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&imgoutBlueBuf);
	err = clSetKernelArg(kernel, 6, sizeof(int), &radius);

	// Enqueue the kernel:
	// in order to not modify the code, we will be creating a thread for each pixel of the image. Therefore, the global size needs to be width * height
	size_t globalSize[2] = {width, height};
	size_t localSize[2] = {5, 50};
	cl_event event;

	err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, &globalSize, &localSize, 0, NULL, &event);

	// Wait for the kernel to finish
	err = clWaitForEvents(1, &event);

	// Calculate the execution time
	cl_ulong startTime, endTime;
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);

	cl_ulong executionTime = endTime - startTime;
	// convert to seconds
	double executionTimeInSeconds = (double) executionTime / 1000000000.0;
	printf("Total execution time (opencl): %.6f seconds\n", executionTimeInSeconds);

	// Read the result
	err = clEnqueueReadBuffer(commandQueue, imgoutRedBuf, CL_TRUE, 0, sizeof(unsigned char) * pixels, imgout->red, 0, NULL, NULL);
	err = clEnqueueReadBuffer(commandQueue, imgoutGreenBuf, CL_TRUE, 0, sizeof(unsigned char) * pixels, imgout->green, 0, NULL, NULL);
	err = clEnqueueReadBuffer(commandQueue, imgoutBlueBuf, CL_TRUE, 0, sizeof(unsigned char) * pixels, imgout->blue, 0, NULL, NULL);

	// Clean up
	clReleaseMemObject(imginRedBuf);
	clReleaseMemObject(imginGreenBuf);
	clReleaseMemObject(imginBlueBuf);
	clReleaseMemObject(imgoutRedBuf);
	clReleaseMemObject(imgoutGreenBuf);
	clReleaseMemObject(imgoutBlueBuf);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);
}


double timeit(void (*func)(), int radius,
    img_t *imgin, img_t *imgout)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	func(radius, imgin, imgout);
	gettimeofday(&end, NULL);
	return (double) (end.tv_usec - start.tv_usec) / 1000000
		+ (double) (end.tv_sec - start.tv_sec);
}


char *remove_ext(char *str, char extsep, char pathsep)
{
	char *newstr, *ext, *lpath;

	if (str == NULL) return NULL;
	if ((newstr = malloc(strlen(str) + 1)) == NULL) return NULL;

	strcpy(newstr, str);
	ext = strrchr(newstr, extsep);
	lpath = (pathsep == 0) ? NULL : strrchr(newstr, pathsep);
	if (ext != NULL)
	{
		if (lpath != NULL)
		{
			if (lpath < ext)
				*ext = '\0';
		}
		else
			*ext = '\0';
	}
	return newstr;
}


int main(int argc, char *argv[])
{
	int i, j, radius;
	double exectime_serial = 0.0, exectime_omp_loops = 0.0, exectime_omp_tasks = 0.0;
	struct timeval start, stop;
	char *inputfile, *noextfname;
	char seqoutfile[128], paroutfile_loops[128], paroutfile_tasks[128], paroutfile_opencl[128];
	img_t imgin, imgout, pimgout_loops, pimgout_tasks, pimgout_opencl;

	if (argc < 3)
	{
		fprintf(stderr, "Syntax: %s <blur-radius> <filename>, \n\te.g. %s 2 500.bmp\n",
			argv[0], argv[0]);
		fprintf(stderr, "Available images: 500.bmp, 1000.bmp, 1500.bmp\n");
		exit(1);
	}

	inputfile = argv[2];

	radius = atoi(argv[1]);
	if (radius < 0)
	{
		fprintf(stderr, "Radius should be an integer >= 0; exiting.");
		exit(1);
	}

	noextfname = remove_ext(inputfile, '.', '/');
	sprintf(seqoutfile, "%s-r%d-serial.bmp", noextfname, radius);
	sprintf(paroutfile_loops, "%s-r%d-omp-loops.bmp", noextfname, radius);
	sprintf(paroutfile_tasks, "%s-r%d-omp-tasks.bmp", noextfname, radius);
	sprintf(paroutfile_opencl, "%s-r%d-opencl.bmp", noextfname, radius);

	bmp_read_img_from_file(inputfile, &imgin);
	bmp_clone_empty_img(&imgin, &imgout);
	bmp_clone_empty_img(&imgin, &pimgout_loops);
	bmp_clone_empty_img(&imgin, &pimgout_tasks);
	bmp_clone_empty_img(&imgin, &pimgout_opencl);

	bmp_rgb_alloc(&imgin);
	bmp_rgb_alloc(&imgout);
	bmp_rgb_alloc(&pimgout_loops);
	bmp_rgb_alloc(&pimgout_tasks);
	bmp_rgb_alloc(&pimgout_opencl);

	printf("<<< Gaussian Blur (h=%d,w=%d,r=%d) >>>\n", imgin.header.height,
	       imgin.header.width, radius);

	/* Image data to R,G,B */
	bmp_rgb_from_data(&imgin);

	/* Run & time serial Gaussian Blur */
	exectime_serial = timeit(gaussian_blur_serial, radius, &imgin, &imgout);

	/* Save the results (serial) */
	bmp_data_from_rgb(&imgout);
	bmp_write_data_to_file(seqoutfile, &imgout);

	/* Run & time OpenMP Gaussian Blur (w/ loops) */
	exectime_omp_loops = timeit(gaussian_blur_omp_loops, radius, &imgin, &pimgout_loops);

	/* Save the results (parallel w/ loops) */
	bmp_data_from_rgb(&pimgout_loops);
	bmp_write_data_to_file(paroutfile_loops, &pimgout_loops);

	/* Run & time OpenMP Gaussian Blur (w/ tasks) */
	exectime_omp_tasks = timeit(gaussian_blur_omp_tasks, radius, &imgin, &pimgout_tasks);

	/* Save the results (parallel w/ tasks) */
	bmp_data_from_rgb(&pimgout_tasks);
	bmp_write_data_to_file(paroutfile_tasks, &pimgout_tasks);

	printf("Total execution time (sequential): %lf\n", exectime_serial);
	printf("Total execution time (omp loops): %lf\n", exectime_omp_loops);
	printf("Total execution time (omp tasks): %lf\n", exectime_omp_tasks);

	gaussian_blur_opencl(radius, &imgin, &pimgout_opencl);
	bmp_data_from_rgb(&pimgout_opencl);
    bmp_write_data_to_file(paroutfile_opencl, &pimgout_opencl);


	bmp_img_free(&imgin);
	bmp_img_free(&imgout);
	bmp_img_free(&pimgout_loops);
	bmp_img_free(&pimgout_tasks);

	return 0;
}
