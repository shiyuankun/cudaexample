#include <stdio.h>
#include "matrix.h"

__global__ void
MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0;
    for (int k = 0;k < M->width;k++)
    {
        Pvalue += M->elements[ty * M->width + k] * N->elements[k * N->width + tx];
    }
    P->elements[ty * P->pitch + tx] = Pvalue;
}

void
testMatrixMulOnDevice()
{
    cudaSetDevice(0);
    Matrix M = Initialize(MIDDLE, HEIGHT, 1);
    Matrix N = Initialize(WIDTH, MIDDLE, 1);
    Matrix P = Initialize(WIDTH, HEIGHT, 0);
    Matrix P2 = Initialize(WIDTH, HEIGHT, 0);
    Matrix Md = InitializeDevice(M);
    Matrix Nd = InitializeDevice(N);
    Matrix Pd = InitializeDevice(P);

    clock_t start, finish;
    double elapsed_time;
    start = clock();
    CopyToDeviceMatrix(Md, M);
    CopyToDeviceMatrix(Nd, N);
    dim3 dimBlock(WIDTH, HEIGHT);
    dim3 dimGrid(1, 1);
    MatrixMulOnDevice<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
    CopyFromDeviceMatrix(P, Pd);
    finish = clock();
    elapsed_time = (float)(finish - start) / (float)CLOCKS_PER_SEC;
    printf("spend on simple device multiplication %f\n", elapsed_time);
    start = clock();
    MatrixMulOnHost(M, N, P2);
    finish = clock();
    elapsed_time = (float)(finish - start) / (float)CLOCKS_PER_SEC;
    printf("spend on simple multiplication %f\n", elapsed_time);
    if (CmpMat(P, P2))
        printf("%s\n", "correct");
    else
        printf("%s\n", "wrong");
    FreeDeviceMatrix(Md);
    FreeDeviceMatrix(Nd);
    FreeDeviceMatrix(Pd);

    FreeMatrix(M);
    FreeMatrix(N);
    FreeMatrix(P);
    FreeMatrix(P2);
    return;
}

int main()
{
    printf("Matrix Multiplication on Host\n");
    testMatrixMulOnDevice();
    return 0;
}

