CFLAGS := -std=c99 -Wall -Wextra
LDFLAGS := -lm -lmpi

all:
	$(shell mpicc -showme) ${CFLAGS} src/mpi_toroid.c -o \
	mpi_toroid ${LDFLAGS}

clean:
	rm -f mpi_toroid