// M*N=P
// M(n,m) M(MIDDLE,HEIGHT)
// N(k,n) N(WIDTH,MIDDLE)
// P(k,m) P(WIDTH,HEIGHT)
typedef struct MatrixStruct *Matrix;
int CmpMat(Matrix l, Matrix r);
Matrix Initialize(int width, int height, int initval);
Matrix InitializeDevice(Matrix Mhost);
Matrix TransposeHost(const Matrix Mhost);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix(Matrix Mdevice);
void FreeMatrix(Matrix Mhost);
void PrintMatrix(Matrix Mhost);
void PrintMatrixDevice(Matrix Mdevice);
void randomseed();
float random(float low, float up);

void MatrixMulOnHost(const Matrix M, const Matrix N, Matrix P);


struct MatrixStruct{
    int width;
    int height;
    int pitch;
    float * elements;
};

#define WIDTH (31) //k
#define MIDDLE (32) //n
#define HEIGHT (33) //m
