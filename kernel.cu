
#define _USE_MATH_DEFINES
#define THREAD_COUNT 30
#define INPUT_IMG "test_image.png"
#define PI 3.14159265358979323846
#define KERNEL 5

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "lodepng.cpp"
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <fstream> 
#include <iostream>

typedef unsigned char byte;

struct Image {
	Image(byte* pixels = nullptr, unsigned int width = 0, unsigned int height = 0)
		: _pixels(pixels), _width(width), _height(height) {};
	byte* _pixels;
	unsigned _width;
	unsigned _height;
	LodePNGColorType _colorType;
	unsigned _bitDepth;
};

Image loadPng(const char* fileName) {
	Image image;
	image._colorType = LCT_GREY;
	image._bitDepth = 8;
	unsigned error = lodepng_decode_file(&image._pixels, &image._width, &image._height, fileName, image._colorType, image._bitDepth);
	if (error) {
		printf("Error loading image: %u: %s\n", error, lodepng_error_text(error));
		exit(2);
	}
	return image;
}

void savePng(char* fileName, std::string appendText, Image output) {
	std::string newName = fileName;
	output._colorType = LCT_GREY;
	output._bitDepth = 8;
	newName = newName.substr(0, newName.rfind("."));
	newName.append("_").append(appendText).append(".png");

	unsigned error = lodepng::encode(newName.c_str(), output._pixels, output._width, output._height, output._colorType, output._bitDepth);
	if (error) {
		printf("Error writing image: %u: %s\n", error, lodepng_error_text(error));
		exit(3);
	}
}

void generateGaussFilter(double gaussKernel[5]) {
	// initialising standard devition to 1.0
	double sigma = 1.0;
	double s = 2.0 * sigma * sigma;
	double r;

	// sum is for normalization
	double sum = 0.0;

	// generate 5x5 kernel
	for (int x = -2; x <= 2; x++) {
		for (int y = -2; y <= 2; y++) {
			r = sqrt(x * x + y * y);
			gaussKernel[x + 2 + (y + 2) * 5] = (exp(-(r * r) / s)) / (PI * s);
			sum += gaussKernel[x + 2 + (y + 2) * 5];
		}
	}

	// normalising the Kernel
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < 5; ++j) {
			gaussKernel[i * 5 + j] /= sum;
		}
	}
}

void gaussianBlurCpu(const byte* original, byte* destination, double* gKernel, const unsigned int width, const unsigned int height) {
	for (int y = 0; y < height - 1; y++)
	{
		for (int x = 0; x < width - 1; x++)
		{
			if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
			{
				double sum = 0;
				for (int i = 0; i < KERNEL; i++)
				{
					for (int j = 0; j < KERNEL; j++)
					{
						int num;
						if (y < 4 || x < 4) {
							num = 20;
						}
						else {
							num = original[(y - 2 + i) * width + (x - 2 + j)];
						}
						sum += num * gKernel[i * 5 + j];
					}
				}
				destination[y * width + x] = round(sum);
			}
		}
	}
}

__device__ double d_gKernel[25];

__global__ void gaussianBlurGpu(const byte* d_sourceImg, byte* d_destinationImg, const unsigned int width, const unsigned int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
	{
		double sum = 0;
		for (int i = 0; i < KERNEL; i++)
		{
			for (int j = 0; j < KERNEL; j++)
			{
				int num;
				if (y < 4 || x < 4) {
					num = 20;
				}
				else {
					num = d_sourceImg[(y - 2 + i) * width + (x - 2 + j)];
				}
				sum += num * d_gKernel[i * KERNEL + j];
			}
		}
		d_destinationImg[y * width + x] = round(sum);
	}
}

void sobelEdgeDetectionCpu(const byte* original, byte* destination, const unsigned int width, const unsigned int height) {
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int dx = (-1 * original[(y - 1) * width + (x - 1)]) + (-2 * original[y * width + (x - 1)]) + (-1 * original[(y + 1) * width + (x - 1)]) +
				(original[(y - 1) * width + (x + 1)]) + (2 * original[y * width + (x + 1)]) + (original[(y + 1) * width + (x + 1)]);
			int dy = (original[(y - 1) * width + (x - 1)]) + (2 * original[(y - 1) * width + x]) + (original[(y - 1) * width + (x + 1)]) +
				(-1 * original[(y + 1) * width + (x - 1)]) + (-2 * original[(y + 1) * width + x]) + (-1 * original[(y + 1) * width + (x + 1)]);
			destination[y * width + x] = sqrt((dx * dx) + (dy * dy));
		}
	}
}

__global__ void sobelEdgeDetectionGpu(const byte* d_sourceImg, byte* d_destinationImg, const unsigned int width, const unsigned int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float sobelX;
	float sobelY;
	if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
		sobelX = (-1 * d_sourceImg[(y - 1) * width + (x - 1)]) + (-2 * d_sourceImg[y * width + (x - 1)]) + (-1 * d_sourceImg[(y + 1) * width + (x - 1)]) +
			(d_sourceImg[(y - 1) * width + (x + 1)]) + (2 * d_sourceImg[y * width + (x + 1)]) + (d_sourceImg[(y + 1) * width + (x + 1)]);

		sobelY = (-1 * d_sourceImg[(y - 1) * width + (x - 1)]) + (-2 * d_sourceImg[(y - 1) * width + x]) + (-1 * d_sourceImg[(y - 1) * width + (x + 1)]) +
			(d_sourceImg[(y + 1) * width + (x - 1)]) + (2 * d_sourceImg[(y + 1) * width + x]) + (d_sourceImg[(y + 1) * width + (x + 1)]);

		d_destinationImg[y * width + x] = sqrt((sobelX * sobelX) + (sobelY * sobelY));
	}
}

int main()
{
	Image testImage = loadPng(INPUT_IMG);
	/*savePng("test_image", "black", testImage);*/

	auto height = testImage._height;
	auto width = testImage._width;

	dim3 threadsPerBlock(THREAD_COUNT, THREAD_COUNT);
	dim3 numberOfBlocks(ceil(width / THREAD_COUNT), ceil(height / THREAD_COUNT));


	double gKernel[25];
	generateGaussFilter(gKernel);

	std::string metricsFileName = "metrics";
	std::string inputImg = INPUT_IMG;
	std::string inputImgName = inputImg.substr(0, inputImg.rfind("."));
	metricsFileName.append("_").append(inputImgName).append(".txt");
	std::ofstream metrics(metricsFileName);
	metrics << "gauss-cpu;gauss-gpu;sobel-cpu;sobel-gpu;" << std::endl;

	for (int i = 0; i < 100; i++) {
		printf("%d.\n", i + 1);
#pragma region Gaussian filter[CPU]

	Image gaussDestinationImageCpu(new byte[width * height], width, height);
	auto gaussStartTimeCpu = std::chrono::high_resolution_clock::now();
	gaussianBlurCpu(testImage._pixels, gaussDestinationImageCpu._pixels, gKernel, width, height);
	auto gaussStopTimeCpu = std::chrono::high_resolution_clock::now();

	auto gaussEllapsedTimeCPU = std::chrono::duration_cast<std::chrono::microseconds>(gaussStopTimeCpu - gaussStartTimeCpu);
	printf("Gaussian blur CPU: %ld ms\n", gaussEllapsedTimeCPU.count() / 1000);
	savePng(INPUT_IMG, "gauss_cpu", gaussDestinationImageCpu);
#pragma endregion

#pragma region Gaussian filter[GPU]
		Image gaussDestinationImageGpu(new byte[width * height], width, height);
	
		byte* d_gaussSource;
		byte* d_gaussDestination;
	
		cudaMalloc((void**)&d_gaussSource, (width * height));
		cudaMalloc((void**)&d_gaussDestination, (width * height));
		cudaMemcpy(d_gaussSource, testImage._pixels, (width * height), cudaMemcpyHostToDevice);
		cudaMemset(d_gaussDestination, 0, (width * height));
		cudaMemcpyToSymbol(d_gKernel, gKernel, sizeof(double) * 25);
	
		cudaEvent_t gaussStart, gaussEnd;
		cudaEventCreate(&gaussStart);
		cudaEventCreate(&gaussEnd);
	
		cudaEventRecord(gaussStart, 0);
		gaussianBlurGpu <<< numberOfBlocks, threadsPerBlock >>> (d_gaussSource, d_gaussDestination, width, height);
		cudaEventRecord(gaussEnd, 0);
	
		cudaMemcpy(gaussDestinationImageGpu._pixels, d_gaussDestination, (width * height), cudaMemcpyDeviceToHost);
	
		cudaEventSynchronize(gaussEnd);
		float gaussElapsedTimeGPU = 0.0f;
		cudaEventElapsedTime(&gaussElapsedTimeGPU, gaussStart, gaussEnd);
		printf("Gaussian blur GPU: %f ms\n", gaussElapsedTimeGPU);
	
		savePng(INPUT_IMG, "gauss_gpu", gaussDestinationImageGpu);
	
		cudaFree(d_gaussSource);
		cudaFree(d_gaussDestination);
	#pragma endregion

#pragma region Sobel edge detection on blurred image[CPU]
		Image sobelDestinationImageCpu(new byte[width * height], width, height);

		auto sobelStartCpu = std::chrono::high_resolution_clock::now();
		sobelEdgeDetectionCpu(gaussDestinationImageCpu._pixels, sobelDestinationImageCpu._pixels, width, height);
		auto sobelEndCpu = std::chrono::high_resolution_clock::now();

		auto sobelElapsedTimeCpu = std::chrono::duration_cast<std::chrono::microseconds>(sobelEndCpu - sobelStartCpu);
		printf("Sobel edge detection CPU: %ld ms\n", sobelElapsedTimeCpu.count() / 1000);
		savePng(INPUT_IMG, "sobel_cpu", sobelDestinationImageCpu);
#pragma endregion

#pragma region Sobel edge detection on blurred image[GPU]
		Image sobelDestinationImageGpu(new byte[width * height], width, height);

		byte* d_sobelSource;
		byte* d_sobelDestination;
		cudaMalloc((void**)&d_sobelSource, (width * height));
		cudaMalloc((void**)&d_sobelDestination, (width * height));
		cudaMemcpy(d_sobelSource, gaussDestinationImageGpu._pixels, (width * height), cudaMemcpyHostToDevice);
		cudaMemset(d_sobelDestination, 0, (width * height));

		cudaEvent_t sobelStart, sobelEnd;
		cudaEventCreate(&sobelStart);
		cudaEventCreate(&sobelEnd);

		cudaEventRecord(sobelStart, 0);
		sobelEdgeDetectionGpu <<< numberOfBlocks, threadsPerBlock >> > (d_sobelSource, d_sobelDestination, width, height);
		cudaEventRecord(sobelEnd, 0);

		cudaMemcpy(sobelDestinationImageGpu._pixels, d_sobelDestination, (width * height), cudaMemcpyDeviceToHost);

		cudaEventSynchronize(sobelEnd);
		float sobelElapsedTimeGpu = 0.0f;
		cudaEventElapsedTime(&sobelElapsedTimeGpu, sobelStart, sobelEnd);
		printf("Sobel edge detection GPU: %f ms\n", sobelElapsedTimeGpu);

		savePng(INPUT_IMG, "sobel_gpu", sobelDestinationImageGpu);

		cudaFree(d_sobelSource);
		cudaFree(d_sobelDestination);
#pragma endregion

		delete[] gaussDestinationImageCpu._pixels;
		delete[] gaussDestinationImageGpu._pixels;
		delete[] sobelDestinationImageCpu._pixels;
		delete[] sobelDestinationImageGpu._pixels;

		std::string line = std::to_string(gaussEllapsedTimeCPU.count() / 1000) + ";" + std::to_string(gaussElapsedTimeGPU) + ";"
			+ std::to_string(sobelElapsedTimeCpu.count() / 1000) + ";" + std::to_string(sobelElapsedTimeGpu);
		metrics << line.c_str() << std::endl;
		}
		metrics.close();

	return 0;
}
