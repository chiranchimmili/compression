#include <cuda_runtime.h>
#include <nvcomp/lz4.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void cpuMatrixMul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matrixMulKernel(float* A_decompressed, float* B_decompressed, float* C, int N, int blockSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A_decompressed[row * N + k] * B_decompressed[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void compressBlockLZ4(float* input, size_t inputSize, void** compressedData, size_t* compressedSize) {
    nvcompLZ4FormatOpts opts;
    size_t tempSize;
    void* temp;
    size_t metadataSize;
    nvcompLZ4CompressGetTempSize(inputSize, &opts, &tempSize);
    cudaMalloc(&temp, tempSize);
    
    nvcompLZ4CompressGetMetadataSize(&opts, &metadataSize);
    cudaMalloc(compressedData, inputSize + metadataSize);
    cudaMalloc(compressedSize, sizeof(size_t));
    
    nvcompLZ4CompressAsync(input, inputSize, temp, tempSize, *compressedData, *compressedSize, &opts, 0, cudaStream_t(0));
    cudaFree(temp);
}

void decompressBlockLZ4(void* compressedData, size_t compressedSize, float** decompressedData, size_t decompressedSize) {
    nvcompLZ4FormatOpts opts;
    size_t tempSize;
    void* temp;
    nvcompLZ4DecompressGetTempSize(compressedData, compressedSize, &opts, &tempSize);
    
    cudaMalloc(&temp, tempSize);
    cudaMalloc(decompressedData, decompressedSize);
    
    nvcompLZ4DecompressAsync(compressedData, compressedSize, temp, tempSize, *decompressedData, decompressedSize, &opts, 0, cudaStream_t(0));
    cudaFree(temp);
}

bool compareMatrices(const float* C_cpu, const float* C_gpu, int N) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-4) {
            return false;
        }
    }
    return true;
}


int main() {
    int N = 512;
    size_t matrixSize = N * N * sizeof(float);
    
    float* h_A = (float*)malloc(matrixSize);
    float* h_B = (float*)malloc(matrixSize);
    float* h_C_cpu = (float*)malloc(matrixSize);
    float* h_C_gpu = (float*)malloc(matrixSize);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    cpuMatrixMul(h_A, h_B, h_C_cpu, N);

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);
    
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    void* d_A_compressed;
    void* d_B_compressed;
    size_t d_A_compressedSize, d_B_compressedSize;
    compressBlockLZ4(d_A, matrixSize, &d_A_compressed, &d_A_compressedSize);
    compressBlockLZ4(d_B, matrixSize, &d_B_compressed, &d_B_compressedSize);

    dim3 dimBlock(32, 32);
    dim3 dimGrid((N + 32 - 1) / 32, (N + 32 - 1) / 32);

    float* d_A_decompressed;
    float* d_B_decompressed;
    decompressBlockLZ4(d_A_compressed, d_A_compressedSize, &d_A_decompressed, matrixSize);
    decompressBlockLZ4(d_B_compressed, d_B_compressedSize, &d_B_decompressed, matrixSize);
    
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A_decompressed, d_B_decompressed, d_C, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C_gpu, d_C, matrixSize, cudaMemcpyDeviceToHost);

    if (compareMatrices(h_C_cpu, h_C_gpu, N)) {
        printf("results match\n");
    } else {
        printf("results do not match\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_compressed);
    cudaFree(d_B_compressed);
    cudaFree(d_A_decompressed);
    cudaFree(d_B_decompressed);

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    return 0;
}