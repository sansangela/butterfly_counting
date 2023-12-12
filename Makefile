CC = gcc
CFLAGS = -O3 -std=c99

all: kernel kernel_parallel

kernel: 
	$(CC) $(CFLAGS) $(OBJS) kernel.c  -o kernel.x -march=native -mfma -mavx
run:
	./kernel.x

kernel_parallel: 
	$(CC) $(CFLAGS) $(OBJS) kernel_parallel.c  -o kernel_parallel.x -march=native -mfma -mavx -fopenmp
run_parallel:
	./kernel_parallel.x

test_parallel: 
	$(CC) $(CFLAGS) $(OBJS) test_parallel.c  -o test_parallel.x -march=native -mfma -mavx -fopenmp
run_test_parallel:
	./test_parallel.x

assemble:
	objdump -s -d -f --source ./kernel.x > kernel.S

clean:
	rm -f *.x *~ *.o *.S