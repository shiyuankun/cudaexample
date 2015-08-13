CC=/usr/local/cuda/bin/nvcc
CFLAGS= -g -arch sm_20

#SOURCES=$(wildcard *.cu)
#OBJS=$(patsubst %.cu,%.cuo,$(SOURCES))
UTILOBJS=util.cu

default:
	make all

all:$(UTILOBJS) MatMulBlk MatMulCudaSimple

MatMulSimple:
	$(CC) $(CFLAGS) $(UTILOBJS) MatMulSimple.cu $< -o MatMulSimple.exe

MatMulBlk:
	$(CC) $(CFLAGS) $(UTILOBJS) MatMulBlock.cu $< -o MatMulBlock.exe

MatMulCudaSimple:
	$(CC) $(CFLAGS) $(UTILOBJS) MatMulSimple.cu MatMulCudaSimple.cu $< -o MatMulCudaSimple.exe


%.exe: %.cu
	$(CC) $(CFLAGS) $(UTILOBJS) $< -o $@
#%.cuo: %.cu
#	$(CC) $(CFLAGS) $< -c -o $@

clean:
	rm -rf *.exe
