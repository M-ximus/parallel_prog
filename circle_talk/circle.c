#include <stdio.h>
#include "mpi.h"

const int message = 0xDED;

int main(int argc, char* argv[])
{
    int error = MPI_Init(NULL, NULL);
    if (error != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Init errror = %d\n", error);
        return -1;
    }

    int num_nodes = 0;
    error = MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    if (error != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Comm_size error = %d", error);
        MPI_Finalize();
        return -1;
    }

    if (num_nodes == 1)
    {
        printf("number = %X\n", message);
        MPI_Finalize();
        return 0;
    }

    int node_number = 0;
    error = MPI_Comm_rank(MPI_COMM_WORLD, &node_number);
    if (error != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Comm_rank error = %d\n", error);
        MPI_Finalize();
        return -1;
    }

    if (node_number == 0)
    {
        error = MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        if (error != MPI_SUCCESS)
        {
            fprintf(stderr, "MPI_Send error = %d\n", error);
            MPI_Finalize();
            return -1;
        }

        int recv_mess = 0;
        error = MPI_Recv(&recv_mess, 1, MPI_INT, num_nodes - 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        if (error != MPI_SUCCESS)
        {
            fprintf(stderr, "MPI_Recv last mess by zero node error = %d\n", error);
            MPI_Finalize();
            return -1;
        }

        printf("number = %X\n", recv_mess);
    }
    else if (node_number == num_nodes - 1)
    {
        int recv_mess = 0;
        error = MPI_Recv(&recv_mess, 1, MPI_INT, node_number - 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        if (error != MPI_SUCCESS)
        {
            fprintf(stderr, "MPI_Recv last node mess error = %d\n", error);
            MPI_Finalize();
            return -1;
        }

        error = MPI_Send(&recv_mess, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (error != MPI_SUCCESS)
        {
            fprintf(stderr, "MPI_Send last node error = %d\n", error);
            MPI_Finalize();
            return -1;
        }
    }
    else
    {
        int recv_mess = 0;
        error = MPI_Recv(&recv_mess, 1, MPI_INT, node_number - 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        if (error != MPI_SUCCESS)
        {
            fprintf(stderr, "MPI_Recv %d node mess error = %d\n", node_number, error);
            MPI_Finalize();
            return -1;
        }

        error = MPI_Send(&recv_mess, 1, MPI_INT, node_number + 1, 0, MPI_COMM_WORLD);
        if (error != MPI_SUCCESS)
        {
            fprintf(stderr, "MPI_Send last node error = %d\n", error);
            MPI_Finalize();
            return -1;
        }
    }

    MPI_Finalize();
    return 0;
}
