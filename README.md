How to run:

- Your device's GPU must be CUDA-capable
- You need nvcc on your computer
- Edit the Makefile so that it uses your nvcc
- Run make
- Run ./main.out

How to verify correctness:

- Open main.cu
- Uncomment "printPosition();" under the "loop" function
- Run make
- Run ./main.out

- Feel free to uncomment different test cases in the main function

How to verify speed:

- You need nvprof
- Run test.sh
- To compare to serial implementation:
  - Open main.cu
  - Uncomment "updateVelocitySerial();" under the "loop" function
  - Comment out "updateVelocityParallel();" under the "loop" function
  - Run make
  - Run ./main.out

