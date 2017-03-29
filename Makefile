all:
	/usr/local/cuda-8.0/bin/nvcc main.cu -o main.out
clean:
	rm main.out

