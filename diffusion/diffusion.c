#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>

enum errors
{
    E_SUCCESS  = 0,
    E_ERROR    = -1,
    E_BADARGS  = -2,
    E_BADALLOC = -3
};

const double tau = 0.001;
const double h   = 0.0001;

const double MAX_TIME  = 2.0;
const double MAX_COORD = 8.0;

const char OUT_FILE[] = "result.dat";

double psi(double t)
{
    return cos(-t * 3.14);
}

double phi(double x)
{
    return cos(x * 3.14);
}

double f(double x, double t)
{
    return 0;
}

int init_coord_row(double* row, double time_step, double time_start, double x_step, double start, double end, int left, int right);
int save_data(double* data, size_t row_size, size_t col_size, const char* file_name);
int calc_all_map(double* map, double time_step, double time_start, double time_end, double x_step, double x_start, double x_end, int left, int right);

int main(int argc, char* argv)
{
    int ret = MPI_Init(NULL, NULL);
    if (ret != MPI_SUCCESS)
    {
        fprintf(stderr, "[main] MPI_Init returned error\n");
        exit(EXIT_FAILURE);
    }

    double start_time = MPI_Wtime();

    int num_nodes;
    ret = MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    if (ret != MPI_SUCCESS)
    {
        fprintf(stderr, "[main] Can't get comm size via MPI_Comm_size - returned %d\n", ret);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (num_nodes == 1)
    {
        size_t num_time_steps  = (size_t)(MAX_TIME / tau);
        size_t num_coord_steps = (size_t)(MAX_COORD / h);

        printf("array size: time_steps = %ld, coord_steps = %ld\n", num_time_steps, num_coord_steps);

        errno = 0;
        double* time_coord_map = (double*) malloc(sizeof(*time_coord_map) * num_time_steps * num_coord_steps);
        if (time_coord_map == NULL)
        {
            perror("[main][single] Alloc result buffer retuned error\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        int ret = init_coord_row(time_coord_map, tau, 0.0, h, 0.0, MAX_COORD, -1, -1);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[main][single] Calc initial coordinate row returned error %d\n", ret);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        ret = calc_all_map(time_coord_map, tau, 0.0, MAX_TIME, h, 0.0, MAX_COORD, -1, -1);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[main][single] Calc all time-coordinate map returned error %d\n", ret);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        /*
        ret = save_data(time_coord_map, num_coord_steps, num_time_steps, OUT_FILE);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[main][single] Save data to file returned error %d\n", ret);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }*/

        double end_time = MPI_Wtime();
        printf("all time = %lg\n", end_time - start_time);
    }
    else
    {
        int node_number = 0;
        int ret = MPI_Comm_rank(MPI_COMM_WORLD, &node_number);
        if (ret != MPI_SUCCESS)
        {
            fprintf(stderr, "[main][multi] MPI_Comm_rank error = %d\n", ret);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        int left_neigh = node_number - 1;
        if (node_number == 0)
            left_neigh = -1;

        int right_neigh = node_number + 1;
        if (node_number == num_nodes - 1)
            right_neigh = -1;

        size_t num_time_steps = (size_t)(MAX_TIME / tau);

        size_t num_coord_steps = (size_t)(MAX_COORD / h) / num_nodes;
        double x_start = node_number * num_coord_steps * h;
        double x_end   = x_start + num_coord_steps * h;
        if (x_end > MAX_COORD)
        {
            x_end = MAX_COORD;
            num_coord_steps = (size_t)((x_end - x_start) / h);
        }

        errno = 0;
        double* time_coord_map = (double*) malloc(sizeof(*time_coord_map) * num_time_steps * num_coord_steps);
        if (time_coord_map == NULL)
        {
            perror("[main][multi] alloc map returned error\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        ret = init_coord_row(time_coord_map, tau, 0, h, x_start, x_end, left_neigh, right_neigh);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[main][multi] node %d - init coord row returned error - %d\n", node_number, ret);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        ret = calc_all_map(time_coord_map, tau, 0, MAX_TIME, h, x_start, x_end, left_neigh, right_neigh);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[main][multi] node %d - calc all map returned error - %d\n", node_number, ret);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        double* all_map;
        if (node_number == 0)
        {
            errno = 0;
            all_map = (double*) calloc(num_time_steps * num_coord_steps * num_nodes, sizeof(*all_map));
            if (all_map == NULL)
            {
                perror("[main][multi] alloc recv buffer returned error\n");
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        }

        //size_t recv_size = ((size_t)(MAX_COORD / h)) * num_time_steps;
        //size_t arr_size  = num_time_steps * num_coord_steps * num_nodes;
        //printf("pointer = %p, arr_size = %ld, recv_size = %ld, send_size = %ld\n", all_map, arr_size, recv_size, num_time_steps * num_coord_steps);

        errno = 0;
        ret = MPI_Gather(time_coord_map, num_time_steps * num_coord_steps, MPI_DOUBLE, all_map, num_time_steps * num_coord_steps, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS)
        {
            fprintf(stderr, "[main][multi] gather all returned error - %d\n", ret);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        if (node_number == 0)
        {
            errno = 0;
            double* out_map = (double*) calloc(num_time_steps * num_coord_steps * num_nodes, sizeof(*all_map));
            if (all_map == NULL)
            {
                perror("[main][multi] alloc recv buffer returned error\n");
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            size_t all_row = num_nodes * num_coord_steps;
            size_t block_size = num_coord_steps * num_time_steps;
            for (int i = 0; i < num_nodes; i++)
            {
                for (size_t time = 0; time < num_time_steps; time++)
                {
                    for (size_t pos = 0; pos < num_coord_steps; pos++)
                    {
                        out_map[time * all_row + i * num_coord_steps + pos] = all_map[i * block_size + time * num_coord_steps + pos];
                    }
                }
            }

            /*
            ret = save_data(out_map, num_coord_steps * num_nodes, num_time_steps, OUT_FILE);
            if (ret != E_SUCCESS)
            {
                fprintf(stderr, "[main][single] Save data to file returned error %d\n", ret);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }*/

            double end_time = MPI_Wtime();
            printf("all time = %lg\n", end_time - start_time);
        }

    }

    MPI_Finalize();
    return 0;
}

int init_coord_row(double* row, double time_step, double time_start, double x_step, double start, double end, int left, int right)
{
    if (row == NULL)
    {
        fprintf(stderr, "[init_coord_row] Bad input row\n");
        return E_BADARGS;
    }

    if (x_step != x_step || start != start || end != end || time_step != time_step || time_start != time_start) // NaN check
    {
        fprintf(stderr, "[init_coord_row] Bad input doubles - %lg, %lg, %lg\n", x_step, start, end);
        return E_BADARGS;
    }

    //printf("I started initializing\n");

    size_t row_size = (size_t)((end - start) / x_step);
    //printf("row_size: row_size = %ld\n", row_size);

    double x = start;
    for (size_t i = 0; x < end && i < row_size; x += x_step, i++)
        row[i] = phi(x);

    double next_time = time_start + time_step;

    if (left == -1)
        row[row_size] = psi(next_time);
    else
    {
        double prev_cell = phi(start - x_step);
        row[row_size] = time_step * (f(start, 0) - (row[1] - prev_cell) / 2 / x_step) + (row[1] + prev_cell) / 2;
    }

    x = start;
    for (size_t i = 1; x < end && i < row_size - 1; x+= x_step, i++)
        row[row_size + i] = time_step * (f(x, 0) - (row[i + 1] - row[i - 1]) / 2 / x_step) + (row[i + 1] + row[i - 1]) / 2;

    if (right == - 1)
    {
        size_t last = row_size - 1;
        double dummy_cell = 2 * row[last] - row[last - 1];
        row[row_size + last] = time_step * (f(end, 0) - (dummy_cell - row[last - 1]) / 2 / x_step) + (dummy_cell + row[last - 1]) / 2;
    }
    else
    {
        double next_cell = phi(end + x_step);
        size_t last = row_size - 1;
        row[row_size + last] = time_step * (f(end, 0) - (next_cell - row[last - 1]) / 2 / x_step) + (next_cell + row[last - 1]) / 2;
    }

    return E_SUCCESS;
}

int calc_all_map(double* map, double time_step, double time_start, double time_end, double x_step, double x_start, double x_end, int left, int right)
{
    if (map == NULL)
    {
        fprintf(stderr, "[calc_all_map] input map array is null\n");
        return E_BADARGS;
    }

    if (time_step != time_step || time_start != time_start || time_end != time_end)
    {
        fprintf(stderr, "[calc_all_map] input time doubles is NaN - %lg, %lg, %lg\n", time_step, time_start, time_end);
        return E_BADARGS;
    }

    if (x_step != x_step || x_start != x_start || x_end != x_end)
    {
        fprintf(stderr, "[calc_all_map] input coordinate doubles is NaN - %lg, %lg, %lg\n", x_step, x_start, x_end);
        return E_BADARGS;
    }

    //printf("I started calculations, left = %d, right = %d\n", left, right);

    size_t row_size = (size_t)((x_end - x_start) / x_step);
    size_t col_size = (size_t)((time_end - time_start) / time_step);

    //printf("array size: row_size = %ld, col_size = %ld\n", row_size, col_size);

    for (size_t row = 1; row < col_size - 1; row++)
    {
        //printf("current row = %ld\n", row);

        size_t next_row = row + 1;
        size_t prev_row = row - 1;

        double time = row * tau + time_start;
        double x    = x_start + x_step;

        MPI_Request left_send_req, left_recv_req;
        MPI_Request right_send_req, right_recv_req;

        double prev_cell, next_cell;
        if (left != -1)
        {
            int err = MPI_Isend(&map[row_size * row], 1, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &left_send_req);
            if (err != MPI_SUCCESS)
            {
                fprintf(stderr, "[calc_all_map] %d node send left border returned error %d", left + 1, err);
                return E_ERROR;
            }

            err = MPI_Irecv(&prev_cell, 1, MPI_DOUBLE, left, MPI_ANY_TAG, MPI_COMM_WORLD, &left_recv_req);
            if (err != MPI_SUCCESS)
            {
                fprintf(stderr, "[calc_all_map] %d node recv left border returned error %d", left + 1, err);
                return E_ERROR;
            }
        }
        if (right != -1)
        {
            int err = MPI_Isend(&map[row_size * row + row_size - 1], 1, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &right_send_req);
            if (err != MPI_SUCCESS)
            {
                fprintf(stderr, "[calc_all_map] %d node send right border returned error %d", right - 1, err);
                return E_ERROR;
            }

            err = MPI_Irecv(&next_cell, 1, MPI_DOUBLE, right, MPI_ANY_TAG, MPI_COMM_WORLD, &right_recv_req);
            if (err != MPI_SUCCESS)
            {
                fprintf(stderr, "[calc_all_map] %d node recv right border returned error %d", right - 1, err);
                return E_ERROR;
            }
        }

        for (size_t col = 1; col < row_size - 1; col++, x+= x_step)
            map[row_size * next_row + col] = time_step * (2 * f(x, time) - (map[row_size * row + (col + 1)] - map[row_size * row + (col - 1)]) / x_step) + map[row_size * prev_row + col];

        //left
        if (left == -1)
            map[row_size * next_row] = psi(time + tau); // left border
        else
        {
            MPI_Status status;
            MPI_Wait(&left_recv_req, &status);

            map[row_size * next_row] = time_step * (2 * f(x_start, time) - (map[row_size * row + 1] - prev_cell) / x_step) + map[row_size * prev_row];

            MPI_Wait(&left_send_req, &status);
        }

        if (right == -1)
        {
            size_t last_col   = row_size - 1;
            double last_x     = x_start + last_col * x_step;
            double dummy_cell = 2 * map[row_size * row + last_col] - map[row_size * row + (last_col - 1)];
            map[row_size * next_row + last_col] = time_step * (2 * f(last_x, time) - (dummy_cell - map[row_size * row + (last_col - 1)]) / x_step) + map[row_size * prev_row + last_col];
        }
        else
        {
            MPI_Status status;
            MPI_Wait(&right_recv_req, &status);

            size_t last_col = row_size - 1;
            double last_x   = x_start + last_col * x_step;
            map[row_size * next_row + last_col] = time_step * (2 * f(last_x, time) - (next_cell - map[row_size * row + (last_col - 1)]) / x_step) + map[row_size * prev_row + last_col];

            MPI_Wait(&right_send_req, &status);
        }
    }

    return E_SUCCESS;
}

int save_data(double* data, size_t row_size, size_t col_size, const char* file_name)
{
    if (data == NULL)
    {
        fprintf(stderr, "[save_data] data array is null pointer\n");
        return E_BADARGS;
    }
    if (file_name == NULL)
    {
        fprintf(stderr, "[save_data] output file ptr is null\n");
        return E_BADARGS;
    }

    errno = 0;
    FILE* out = fopen(file_name, "w");
    if (out == NULL)
    {
        perror("[save_data] open output file returned error\n");
        return E_ERROR;
    }

    //fprintf(out, "%ld\n%ld\n", row_size, col_size);

    for (size_t row = 0; row < col_size; row++)
    {
        for (size_t col = 0; col < row_size; col++)
            fprintf(out, "%lg ", data[row * row_size + col]);
        fprintf(out, "\n");
    }

    fclose(out);

    return E_SUCCESS;
}
