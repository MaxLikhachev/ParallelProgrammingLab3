
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;

#define BLOCK_SIZE 8
#define N 32

const float degreesToRadiansCoefficient = 0.0174533;
const int minValue = 0;
const int maxValue = 360;

/* TODO:
* fix shared memory
* try to fix constant memory
*/

cudaError_t calculateWithCuda(float*c, float*a, float*b, unsigned int size);

__global__ void globalCalculateKernel(float*c, float*a, float*b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    c[i * j] = sin(a[i * j]) * sin(a[i * j]) + cos(b[i * j]) * cos(b[i * j]) * cos(b[i * j]);
}

__global__ void sharedCalculateKernel(float* c, float* a, float* b, unsigned int size)
{
    __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];
    float shared_c_temp= 0;

    for (int i = size * BLOCK_SIZE * blockIdx.y; i <= size * BLOCK_SIZE * blockIdx.y + size - 1; i += BLOCK_SIZE)
    {
        shared_a[threadIdx.x][threadIdx.y] = a[i + size * threadIdx.y + threadIdx.x];
        shared_b[threadIdx.x][threadIdx.y] = b[i + size * threadIdx.y + threadIdx.x];
        shared_c_temp = shared_a[threadIdx.x][threadIdx.y] * shared_a[threadIdx.x][threadIdx.y];
    }
    c[size * BLOCK_SIZE * blockIdx.y + size * BLOCK_SIZE * blockIdx.y + threadIdx.y * size + threadIdx.x] = shared_c_temp;
}
/*
__constant__ float constant_a[N * N];
__constant__ float constant_b[N * N];
__global__ void constantCalculateKernel(float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    c[i * j] = sin(constant_a[i * j]) * sin(constant_a[i * j]) + cos(constant_b[i * j]) * cos(constant_b[i * j]) * cos(constant_b[i * j]);
}
*/
bool isCalculationCorrect(int arraySize, float* c, const float* a, const float* b)
{
    bool isError = true;
    for (int i = 0; i < arraySize && isError; i++)
        for (int j = 0; j < arraySize && isError; j++)
            isError = c[i * j] != sin(a[i * j]) * sin(a[i * j]) + cos(b[i * j]) * cos(b[i * j]) * cos(b[i * j]);
    return isError;
}

void initRandom(int arraySize, float* a) 
{
    for (int i = 0; i < arraySize; i++)
        for (int j = 0; j < arraySize; j++)
            a[i * arraySize + j] = minValue + rand() % maxValue * degreesToRadiansCoefficient;

}

void initNull(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
        for (int j = 0; j < arraySize; j++)
            a[i * arraySize + j] = 0;
}

void display(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j++)
            cout << a[i * arraySize + j] << " ";
        cout << endl;
    }
}

int main()
{
    srand(time(NULL));

    cout << "Enter array size: ";
    int arraySize = 0;
    cin >> arraySize;

    float* a = new float[arraySize * arraySize];
    float* b = new float[arraySize * arraySize];
    float* c = new float[arraySize * arraySize];

    initRandom(arraySize, a);
    initRandom(arraySize, b);
    initNull(arraySize, c);

    // Add matrixes in parallel.
    cudaError_t cudaStatus = calculateWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        cout << "calculateWithCuda failed!\n";
       return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cout << "cudaDeviceReset failed!\n";
        return 1;
    }

    return 0;
}


// Helper function for using CUDA to add matrixes in parallel.
cudaError_t calculateWithCuda(float*c, float*a, float*b, unsigned int size)
{
    float* dev_a;
    float* dev_b;
    float* dev_c;

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float KernelTime;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three matrixes (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, (N * N) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, (N * N) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, (N * N) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input matrixes from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.

    int numBlocks = BLOCK_SIZE;
    dim3 threadsPerBlock(N, N);

    cout << "Config settings: arraySize = " << size << ", numBlocks = " << numBlocks << ", threadsPerBlock(" << N << ", " << N << ")\n";


    // Global memory
    cudaEventRecord(start, 0);
    globalCalculateKernel <<<numBlocks, threadsPerBlock>>> (dev_c, dev_a, dev_b);
    // if (!isCalculationCorrect(size, dev_c, dev_a, dev_b)) cout << "Calculation Error\n";
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "\nGlobal result: " << KernelTime <<  " milliseconds\n";

    // Shared memory
    cudaEventRecord(start, 0);
    sharedCalculateKernel << <numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);
    // if (!isCalculationCorrect(size, dev_c, dev_a, dev_b)) cout << "Calculation Error\n";
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "\Shared result: " << KernelTime << " milliseconds\n";

    // Constant memory
    /*
    cudaEventRecord(start, 0);
    float constant_a[N * N];
    float constant_b[N * N];
    cudaStatus = cudaMemcpy(constant_a, a, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(constant_b, b, (N * N) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    constantCalculateKernel << <numBlocks, threadsPerBlock >> > (dev_c);
    if (!isCalculationCorrect(size, dev_c, dev_a, dev_b)) cout << "Calculation Error\n";
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    cout << "\nConstant result: " << KernelTime << " milliseconds\n";
*/
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "globalCalculateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching globalCalculateKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output matrix from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, (N * N) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
