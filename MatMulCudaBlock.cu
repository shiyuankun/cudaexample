#include <stdio.h>
#include "matrix.h"

#define BLOCK_SIZE (16)

__device__ float
GetMatrixElement(const Matrix Md, int x, int y)
{
    return Md->elements[y * Md->pitch + x];
}

__device__ void
SetMatrixElement(Matrix Md, int x, int y, float value)
{
    Md->elements[y * Md->pitch + x] = value;
}

__device__ Matrix
GetSubMatrix(Matrix Md, int x, int y)
{
    MatrixStruct Mdsub;
    Mdsub.width = BLOCK_SIZE;
    Mdsub.height = BLOCK_SIZE;
    Mdsub.pitch = Md->pitch;
    Mdsub.elements = &Md->elements[Md->pitch * BLOCK_SIZE * y + BLOCK_SIZE * x];
    return &Mdsub;
}

__global__ void
MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int y = by * BLOCK_SIZE + ty;
    int x = bx * BLOCK_SIZE + tx;
        
    float Pvalue = 0.0f;
    
    for (int m = 0; m < (M->width - 1) / BLOCK_SIZE + 1; m++ )
    {
        Matrix Msub = GetSubMatrix(M, m, by);
        Matrix Nsub = GetSubMatrix(N, bx, m);

        __shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];
        
        if (y < M->height && m * BLOCK_SIZE + tx < M->width)
        {
            Ms[tx][ty] = GetMatrixElement(Msub, tx, ty);
        }
        else
            Ms[tx][ty] = 0.0f;
        if (x < N->width && m * BLOCK_SIZE + ty < N->height)
        {
            Ns[tx][ty] = GetMatrixElement(Nsub, tx, ty);
        }
        else
            Ns[tx][ty] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0;k < BLOCK_SIZE;k++ )
        {
            Pvalue += Ms[tx][k] * Ns[k][ty];
        }
        
        __syncthreads();
    }
    Matrix Psub = GetSubMatrix(P, bx, by);
    if (y < P->height && x < P->width)
    {
        SetMatrixElement(Psub, tx, ty, Pvalue);
    }
}


Matrix testMatrixMulOnDevice()
{
    cudaSetDevice(1);
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
    /* printf("Matrix M:\n"); */
    /* printmatrix(M); */
    /* printf("Matrix N:\n"); */
    /* printmatrix(N); */

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((P->width - 1) / dimBlock.x + 1, (P->height - 1 )/ dimBlock.y + 1);
    /* matrixMultiply<<<dimGrid, dimBlock>>>(Md.elements, Nd.elements, Pd.elements, Md.height, Md.width, Nd.height, Nd.width, Pd.height, Pd.width); */
    MatrixMulOnDevice<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

    CopyFromDeviceMatrix(P, Pd);
    finish = clock();
    elapsed_time = (float)(finish - start) / (float)CLOCKS_PER_SEC;
    printf("spend on device block multiplication %f\n", elapsed_time);

    start = clock();
    MatrixMulOnHost(M, N, P2);
    finish = clock();
    elapsed_time = (float)(finish - start) / (float)CLOCKS_PER_SEC;
    printf("spend on simple multiplication %f\n", elapsed_time);

    printf("result check: ");
    if (CmpMat(P, P2))
        printf("\x1B[32m%s\x1B[0m\n", "correct");
    else
        printf("\x1B[31m%s\x1B[0m\n", "wrong");
    /* printf("Matrix P:\n"); */
    /* printmatrix(P); */
    FreeDeviceMatrix(Md);
    FreeDeviceMatrix(Nd);
    FreeDeviceMatrix(Pd);

    FreeMatrix(M);
    FreeMatrix(N);
    FreeMatrix(P);
    FreeMatrix(P2);
    return P;
}

int main()
{
    /* printf("Matrix Multiplication on Host\n"); */
    /* Matrix h = testMatrixMulOnHost(); */
    printf("Matrix Multiplication on Device\n");
    testMatrixMulOnDevice();
    return 0;
}
