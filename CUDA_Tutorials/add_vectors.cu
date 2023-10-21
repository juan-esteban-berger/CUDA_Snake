#include <iostream>
#include <cuda_runtime.h>

__global__ void addVectors(float* vec1, float* vec2, float* result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        result[index] = vec1[index] + vec2[index];
    }
}

int main() {
    const int N = 256;
    float *vec1, *vec2, *result;
    float *d_vec1, *d_vec2, *d_result;

    vec1 = new float[N];
    vec2 = new float[N];
    result = new float[N];

    for (int i = 0; i < N; i++) {
        vec1[i] = float(i);
        vec2[i] = float(i * 2);
    }

    cudaMalloc(&d_vec1, N * sizeof(float));
    cudaMalloc(&d_vec2, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    cudaMemcpy(d_vec1, vec1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, vec2, N * sizeof(float), cudaMemcpyHostToDevice);

    addVectors<<<(N+255)/256, 256>>>(d_vec1, d_vec2, d_result, N);

    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << vec1[i] << " + " << vec2[i] << " = " << result[i] << std::endl;
    }

    delete[] vec1;
    delete[] vec2;
    delete[] result;
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_result);
}

