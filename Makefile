.PHONY: all clean

NVCC := /usr/local/cuda/bin/nvcc

all: test

clean:
	rm -f test

test: test.cu cuda_check.h
	${NVCC} -std=c++14 -O2 -g $< -o $@
