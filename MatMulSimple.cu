#include <stdio.h>
#include <time.h>
#include "matrix.h"

void
MatrixMulOnHost(const Matrix M, const Matrix N, Matrix P)
{
    // simple matrix multiplication equals operatior %*% in R
    // M(MIDDLE,HEIGHT)*N(WIDTH,MIDDLE)=P(WIDTH,HEIGHT)
    // M->Width=N->Height=MIDDLE
    // M->Height=HEIGHT
    // N->Width=WIDTH
    int i, j, k;
    float sum;
    for (i = 0;i < M->height;i++)
        for (j = 0;j < N->width;j++)
        {
            sum = 0.0;
            for (k = 0;k < M->width;k++)
            {
                sum += M->elements[i * M->width + k] * N->elements[k * N->width + j];
            }
            P->elements[i * P->width + j] = sum;
        }
}

void
MatrixMulOnHostT(const Matrix M, const Matrix N, Matrix P)
{
    // transpose N before multiplication to increase N cache hit rate
    // M(MIDDLE,HEIGHT)*N(MIDDLE,WIDTH)=P(WIDTH,HEIGHT)
    // M->Width=N->Width
    // M->Height=HEIGHT
    // N->Height=WIDTH
    int i, j, k;
    float sum;
    for (i = 0;i < M->height;i++)
        for (j = 0;j < N->height;j++)
        {
            sum = 0.0;
            for (k = 0;k < N->width;k++)
            {
                sum += M->elements[i * M->width + k] * N->elements[j * N->width + k];
            }
            P->elements[i * P->width + j] = sum;
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
    printf("M\n");
    PrintMatrix(M);
    printf("N\n");
    PrintMatrix(N);
    start = clock();
    MatrixMulOnHost(M, N, P);
    printf("P\n");
    PrintMatrix(P);
    finish = clock();
    double elapsed_time = finish - start;
    printf("spend on simple multiplication %f\n", elapsed_time);
    start = clock();
    Matrix X = TransposeHost(N);
    MatrixMulOnHostT(M, X, Pt);
    finish = clock();
    elapsed_time = finish - start;
    printf("spend on transpose %f\n", elapsed_time);
    if (CmpMat(P, Pt))
        printf("%s\n", "correct");
    else
        printf("%s\n", "wrong");
    FreeMatrix(M);
    FreeMatrix(N);
    FreeMatrix(P);
    FreeMatrix(Pt);
}

/* int main() */
/* { */
/*     printf("Matrix Multiplication on Host\n"); */
/*     testMatrixMulOnHost(); */
/*     return 0; */
/* } */
