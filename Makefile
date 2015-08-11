CC=/usr/local/cuda/bin/nvcc
CFLAGS= -g

#SOURCES=$(wildcard *.cu)
#OBJS=$(patsubst %.cu,%.cuo,$(SOURCES))
UTILOBJS=util.cu

default:
	make all

all:$(UTILOBJS) MatMulSimple MatMulBlk

MatMulSimple:
	$(CC) $(CFLAGS) $(UTILOBJS) MatMulSimple.cu $< -o MatMulSimple.exe

MatMulBlk:
	$(CC) $(CFLAGS) $(UTILOBJS) MatMulBlock.cu $< -o MatMulBlock.exe

%.exe: %.cu
	$(CC) $(CFLAGS) $(UTILOBJS) $< -o $@
#%.cuo: %.cu
#	$(CC) $(CFLAGS) $< -c -o $@

clean:
	rm -rf *.exe
