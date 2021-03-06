#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

int
CmpMat(Matrix l, Matrix r)
{
    int i, j;
    if (l == NULL || r == NULL)
    {
        printf("NULL pointer\n");
        return 0;
    }
    
    if (l->height != r->height)
        return 0;
    if (l->width != r->width)
        return 0;
    
    for (i = 0;i<l->height;i ++ )
        for (j = 0;j<l->width;j ++ )
            if (fabs(l->elements[i*l->width + j] - r->elements[i*r->width + j])>0.00001 * fabs(l->elements[i*l->width + j]))
            {
                return 0;
            }
    return 1;
}

void
randomseed()
{
    srand((unsigned) time(NULL));
}

float
random(float low, float up)
{
    return (float) rand() / (float) RAND_MAX / (up - low) + low;
}

Matrix
Initialize(int width, int height, int initval)
{
    Matrix x = (Matrix)malloc(sizeof(struct MatrixStruct));
    int i, j;
    x->width = width;
    x->height = height;
    x->pitch = width;
    x->elements = (float *)malloc(width * height * sizeof(float));
    if (initval)
    {
        for (i = 0;i < x->height;i++)
            for (j = 0;j < x->width;j++)
                // x->elements[i * x->width + j] = i*10.0 + j;
                x->elements[i * x->width + j] = random(0.0, 1.0);
    }
    return x;
}

void
CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost->width * Mhost->height * sizeof(float);
    CudaCall(cudaMemcpy(Mdevice->elements, Mhost->elements, size, cudaMemcpyHostToDevice));
}

void
CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice->width * Mdevice->height * sizeof(float);
    CudaCall(cudaMemcpy(Mhost->elements, Mdevice->elements, size, cudaMemcpyDeviceToHost));
}

void
FreeDeviceMatrix(Matrix Mdevice)
{
    CudaCall(cudaFree(Mdevice->elements));
    CudaCall(cudaFreeHost(Mdevice));
}

void
FreeMatrix(Matrix Mhost)
{
    free(Mhost->elements);
    free(Mhost);
}


Matrix
InitializeDevice(Matrix Mhost)
{
    Matrix Mdevice;
    CudaCall(cudaMallocHost((void **)&Mdevice, sizeof(struct MatrixStruct)));
    // I spent a whole day on this sentence
    // the address from malloc cannot be visit by device
    // to allocate memory on host which can be visit by device
    // you have to choose cudaMallocHost
    Mdevice->width = Mhost->width;
    Mdevice->height = Mhost->height;
    Mdevice->pitch = Mhost->pitch;
    int size = Mhost->width * Mhost->height * sizeof(float);
    CudaCall(cudaMalloc((void **)&Mdevice->elements, size));
    return Mdevice;
}

Matrix
TransposeHost(const Matrix Mhost)
{
    int i, j;
    Matrix T = Initialize(Mhost->height, Mhost->width, 0);
    
    for (i = 0;i < Mhost->height;i++)
        for (j = 0;j < Mhost->width;j++)
            T->elements[j * T->width + i] = Mhost->elements[i * Mhost->width + j];
    return T;
}

void
CudaCall(cudaError signal)
{
    if (signal != cudaSuccess)
    {
        printf("Error in cuda call");
    }
}

void
PrintMatrix(Matrix Mhost)
{
    int i, j;
    for (i = 0;i < Mhost->height;i++)
        for (j=0;j < Mhost->width;j++)
        {
            printf("%f%s", Mhost->elements[i * Mhost->width + j], j == Mhost->width - 1 ? "\n" : " ");
        }
}

void
PrintMatrixDevice(Matrix Mdevice)
{
    Matrix T = Initialize(Mdevice->width, Mdevice->height, 0);
    CopyFromDeviceMatrix(T, Mdevice);
    PrintMatrix(T);
    free(T->elements);
    free(T);
}

