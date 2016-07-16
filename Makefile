CFLAGS=-g -fPIC -m64 -Wall
LFLAGS=-fPIC -m64 -Wall -framework opencl
CC=clang

all: main

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o $@

main: main.o
	$(CC) $(LFLAGS) main.o -o main

clean:
	rm main
	rm main.o
