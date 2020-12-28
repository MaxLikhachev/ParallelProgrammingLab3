
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

const float degreesToRadiansCoefficient = 0.0174533;
const int minValue = 0;
const int maxValue = 360;

cudaError_t addWithCuda(float*c, float*a, float*b, unsigned int size);

__global__ void addKernel(float*c, const float*a, const float*b)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    c[i * j] = a[i * j] + b[i * j];
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cout << "Enter array size: ";
    int arraySize = 0;
    cin >> arraySize;
    cout <<  " Array size: " << arraySize << endl;
    //const float a[arraySize][arraySize] = {{ 1, 2, 3, 4, 5 }, { 1, 2, 3, 4, 5 }, { 1, 2, 3, 4, 5 }, { 1, 2, 3, 4, 5 }, { 1, 2, 3, 4, 5 }};
    //const float b[arraySize][arraySize] = { { 10, 20, 30, 40, 50 }, { 10, 20, 30, 40, 50 },{ 10, 20, 30, 40, 50 },{ 10, 20, 30, 40, 50 },{ 10, 20, 30, 40, 50 }, };
    //float c[arraySize][arraySize] = { {0} };

    float* a = new float[arraySize * arraySize];
    float* b = new float[arraySize * arraySize];
    float* c = new float[arraySize * arraySize];

    initRandom(arraySize, a);
    initRandom(arraySize, b);
    initNull(arraySize, c);

    // cout << "A\n";
    // display(arraySize, a);
    // cout << "B\n";
    // display(arraySize, b);
    // cout << "C\n";
    // display(arraySize, c);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        cout << "addWithCuda failed!\n";
       return 1;
    }

    // cout << c[0][0] << c[0][1] << c[0][2] << c[0][3] << c[0][4];
    display(arraySize, c);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cout << "cudaDeviceReset failed!\n";
        return 1;
    }

    return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float*c, float*a, float*b, unsigned int size)
{
    float* dev_a;
    float* dev_b;
    float* dev_c;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, (size * size) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, (size * size) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, (size * size) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, (size * size) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, (size * size) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.

    int numBlocks = 1;
    dim3 threadsPerBlock(size, size);
    addKernel <<<numBlocks, threadsPerBlock>>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, (size * size) * sizeof(float), cudaMemcpyDeviceToHost);
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
