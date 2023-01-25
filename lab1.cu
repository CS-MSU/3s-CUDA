#include <stdio.h>
#include <stdlib.h>

__global__ void subtraction(double *arr_a, double *arr_b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n)
    {
        arr_a[idx] = arr_a[idx] - arr_b[idx];
        idx += offset;
    }
}

int main()
{
    int i;
    unsigned int n;

    fscanf(stdin, "%u", &n);
    double *a = (double *)malloc(sizeof(double) * n);
    for (i = 0; i < n; i++)
    {
        fscanf(stdin, "%lf", &a[i]);
    }
    double *b = (double *)malloc(sizeof(double) * n);
    for (i = 0; i < n; i++)
    {
        fscanf(stdin, "%lf", &b[i]);
    }

    double *cuda_a;
    double *cuda_b;

    cudaMalloc(&cuda_a, sizeof(double) * n);
    cudaMalloc(&cuda_b, sizeof(double) * n);

    cudaMemcpy(cuda_a, a, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(double) * n, cudaMemcpyHostToDevice);

    subtraction<<<1024, 1024>>>(cuda_a, cuda_b, n);

    cudaMemcpy(a, cuda_a, sizeof(double) * n, cudaMemcpyDeviceToHost);

    for (i = 0; i < n; i++)
    {
        fprintf(stdout, "%.10lf ", a[i]);
    }

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    free(a);
    free(b);

    return 0;
}
