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
void randomseed();
float random(float low, float up);

struct MatrixStruct{
    int width;
    int height;
    int pitch;
    float * elements;
};

#define WIDTH (2100) //k
#define MIDDLE (2200) //n
#define HEIGHT (2300) //m
