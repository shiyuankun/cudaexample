#include <stdio.h>
#include <time.h>
#include "matrix.h"

#define STEP_WIDTH (16)
#define STEP_MIDDLE (16)
#define STEP_HEIGHT (16)
void
MatMulBlkOnHost(const Matrix M, const Matrix N, Matrix P)
{
    // matrix multiplication by blocks
    // M(MIDDLE,HEIGHT)*N(WIDTH,MIDDLE)=P(WIDTH,HEIGHT)
    // M->Width=N->Height=MIDDLE
    // M->Height=HEIGHT
    // N->Width=WIDTH
    int i, j, k;
    int i0, j0, k0;
    for (i0 = 0;i0 < M->height;i0+= STEP_HEIGHT)
    {
        for (j0 = 0;j0 < N->width;j0+= STEP_WIDTH)
        {
            for (k0 = 0;k0 < M->width;k0+= STEP_MIDDLE)
            {
                for (i = i0;i < min(i0 + STEP_HEIGHT, M->height);i++)
                {
                    for (j = j0;j < min(j0 + STEP_WIDTH, N->width);j++)
                    {
                        for (k = k0;k < min(k0 + STEP_MIDDLE, M->width);k++)
                        {
                            P->elements[i * P->width + j] += M->elements[i * M->width + k] * N->elements[k * N->width + j];
                        }
                    }
                }
            }
        }
    }
}

void
MatMulBlkOnHostT(const Matrix M, const Matrix N, Matrix P)
{
    // transpose N before multiplication to increase N cache hit rate
    // M(MIDDLE,HEIGHT)*N(MIDDLE,WIDTH)=P(WIDTH,HEIGHT)
    // M->Width=N->Width
    // M->Height=HEIGHT
    // N->Height=WIDTH
    int i, j, k;
    int i0, j0, k0;
    for (i0 = 0;i0 < M->height;i0+= STEP_HEIGHT)
    {
        for (j0 = 0;j0 < N->height;j0+= STEP_WIDTH)
        {
            for (k0 = 0;k0 < N->width;k0+= STEP_MIDDLE)
            {
                for (i = i0;i < min(i0 + STEP_HEIGHT, M->height);i++)
                {
                    for (j = j0;j < min(j0 + STEP_WIDTH, N->height);j++)
                    {
                        for (k = k0;k < min(k0 + STEP_MIDDLE, N->width);k++)
                        {
                            P->elements[i * P->width + j] += M->elements[i * M->width + k] * N->elements[j * N->width + k];
                        }
                    }
                }
            }
        }
    }
}

void
testMatrixMulOnHost()
{
    Matrix M = Initialize(MIDDLE, HEIGHT, 1);
    Matrix N = Initialize(WIDTH, MIDDLE, 1);
    Matrix P = Initialize(WIDTH, HEIGHT, 0);
    Matrix Pt = Initialize(WIDTH, HEIGHT, 0);
    clock_t start, finish;
    start = clock();
    MatMulBlkOnHost(M, N, P);
    finish = clock();
    double elapsed_time = finish - start;
    printf("spend on simple multiplication %f\n", elapsed_time);

    start = clock();
    Matrix X = TransposeHost(N);
    MatMulBlkOnHostT(M, X, Pt);
    finish = clock();
    elapsed_time = finish - start;
    printf("spend on transpose %f\n", elapsed_time);
    if (CmpMat(P, Pt))
        printf("%s\n", "correct");
    else
        printf("%s\n", "wrong");

    FreeMatrix(M);
    FreeMatrix(N);
    FreeMatrix(Pt);
    FreeMatrix(P);
    return;
}



int main()
{
    printf("Matrix Multiplication on Host\n");
    testMatrixMulOnHost();
    return 0;
}
