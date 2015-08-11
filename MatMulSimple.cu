#include <stdio.h>
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
    double sum;
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
    double sum;
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

Matrix
testMatrixMulOnHost()
{
    Matrix M = Initialize(MIDDLE, HEIGHT, 1);
    Matrix N = Initialize(WIDTH, MIDDLE, 1);
    Matrix X = TransposeHost(N);
    Matrix P = Initialize(WIDTH, HEIGHT, 0);
    Matrix Pt = Initialize(WIDTH, HEIGHT, 0);
    /* printf("Matrix M:\n"); */
    /* PrintMatrix(M); */
    /* printf("Matrix N:\n"); */
    /* PrintMatrix(N); */
    /* printf("Matrix Transpose of N:\n"); */
    /* PrintMatrix(X); */
    MatrixMulOnHost(M, N, P);
    /* printf("Matrix P:\n"); */
    /* PrintMatrix(P); */
    /* printf("Matrix P from tranpose N:\n"); */
    MatrixMulOnHostT(M, X, Pt);
    if (CmpMat(P, Pt))
        printf("%s\n", "correct");
    else
        printf("%s\n", "wrong");
    /* PrintMatrix(Pt); */
    FreeMatrix(M);
    FreeMatrix(N);
    FreeMatrix(Pt);
    return P;
// FreeMatrix(P);
}



int main()
{
    printf("Matrix Multiplication on Host\n");
    Matrix h = testMatrixMulOnHost();
    /* printf("Matrix Multiplication on Device\n"); */
    /* Matrix d = testMatrixMulOnDevice(); */
    /* printf("result check: "); */
    /* if (cmp(h, d)) */
    /*     printf("%s\n", "correct"); */
    /* else */
    /*     printf("%s\n", "wrong"); */
    return 0;
}
