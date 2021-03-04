#include <stdio.h>
#include "mpi.h"

const int NUM_OF_MEMBERS = 1000000;

int calc_garm(int node_num, int num_nodes, double* ret)
{
    if (node_num < 0 || num_nodes < 0 || ret == NULL)
    {
        fprintf(stderr, "calc_garm bad args: %d %d %p", node_num, num_nodes, ret);
        MPI_Finalize();
        return -1;
    }

    int range_size = NUM_OF_MEMBERS / num_nodes;

    int counter = range_size * node_num + 1;
    int end   = counter + range_size;
    if (end > NUM_OF_MEMBERS)
        end = NUM_OF_MEMBERS;

    double sum = 0.0;
    for (;counter < end; counter++)
        sum += 1.0 / (double)(counter);

    *ret = sum;

    return 0;
}

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
//------------------------------------------------------------------------------
    if (num_nodes == 1)
    {
        double sum = 0.0;
        error = calc_garm(0, 1, &sum);
        if (error != 0)
        {
            fprintf(stderr, "calc by 1 node error\n");
            MPI_Finalize();
            return -1;
        }

        printf("sum = %lg\n", sum);

        MPI_Finalize();
        return 0;
    }
//------------------------------------------------------------------------------
    int node_number = 0;
    error = MPI_Comm_rank(MPI_COMM_WORLD, &node_number);
    if (error != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Comm_rank error = %d\n", error);
        MPI_Finalize();
        return -1;
    }

    double sum = 0.0;
    error = calc_garm(node_number, num_nodes , &sum);
    if (error != 0)
    {
        fprintf(stderr, "calc_garm on %d node return error\n", node_number);
        MPI_Finalize();
        return -1;
    }

    if (node_number == 0)
    {
        for (int i = 0; i < num_nodes - 1; i++)
        {
            double recv_sum = 0.0;
            error = MPI_Recv(&recv_sum, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
            if (error != MPI_SUCCESS)
            {
                fprintf(stderr, "MPI_Recv by zero node error = %d\n", error);
                MPI_Finalize();
                return -1;
            }

            sum += recv_sum;
        }

        printf("sum = %lg\n", sum);
    }
    else
    {
        error = MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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
