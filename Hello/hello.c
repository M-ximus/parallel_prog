#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
    int MyRank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank( MPI_COMM_WORLD, &MyRank);
    printf("Hello world, I'm %d\n", MyRank);

    MPI_Finalize();

    return 0;
}
