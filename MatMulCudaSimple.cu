#include <stdio.h>
#include "matrix.h"

__global__ void
MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0;
    for (int k = 0;k<M->width; ++ k)
    {
        float Melement = M.elements[ty * M.width + k];
        float Nelement = N.elements[k * N.width + tx];
        Pvalue += Melement * Nelement;
    }
    P.elements[ty * P.pitch + tx] = Pvalue;
}

Matrix testMatrixMulOnDevice()
{
    cudaSetDevice(1);
    Matrix M = AllocateMatrix(WIDTH, HEIGHT, 1);
    Matrix N = AllocateMatrix(HEIGHT, WIDTH, 1);
    Matrix P = AllocateMatrix(HEIGHT, HEIGHT, 0);
    Matrix Md = AllocateDeviceMatrix(M);
    Matrix Nd = AllocateDeviceMatrix(N);
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Md, M);
    CopyToDeviceMatrix(Nd, N);
    CopyToDeviceMatrix(Pd, P);
    printf("Matrix M:\n");
    printmatrix(M);
    printf("Matrix N:\n");
    printmatrix(N);
    dim3 dimBlock(HEIGHT, HEIGHT);
    dim3 dimGrid(1, 1);
    MatrixMulOnDevice<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

    CopyFromDeviceMatrix(P, Pd);
    
    printf("Matrix P:\n");
    printmatrix(P);
    FreeDeviceMatrix(Md);
    FreeDeviceMatrix(Nd);
    FreeDeviceMatrix(Pd);

    FreeMatrix(M);
    FreeMatrix(N);
    return P;
    /* FreeMatrix(P); */
}


