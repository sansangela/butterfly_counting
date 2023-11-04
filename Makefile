CC = gcc
CFLAGS = -O3 -std=c99

all: kernel

kernel: 
	$(CC) $(CFLAGS) $(OBJS) kernel.c  -o kernel.x -march=native -mfma -mavx
run:
	./kernel.x

clean:
	rm -f *.x *~ *.o
