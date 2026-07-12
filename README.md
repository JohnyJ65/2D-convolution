# 2D Convolution
The following program is an optimised convolution for image processing. Implmentations of it could include sharpening, blurring, or any effects for manipulating a pixel based on the 24 that surround it in a 5x5 square radius through multiplication. 

This convolution uses a double buffered tiled shared memory implementation (20x20) to hide latency, the CUDA pipline API, and memcpyasync API to asychronously load data while the math cores are manipulating the current tile. It also stores the filter size (5x5) in constant memory to improve performance.

The test case uses a standard blur filter to demonstrate the kernel, but as mentioned before, it can be adapted to manipulate the data in other ways. 

Requires Ampere arcitecture or newer (eg. 3060), and must be compiled with sm_70 or newer.
Example compile:

nvcc -O3 -arch=sm_86 2D-convolution.cu -o 2D-convolution

Run:

./2D-convolution# 2D-convolution
