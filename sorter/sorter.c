#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <mpi.h>

enum error
{
    E_SUCCESS = 0,
    E_ERROR = -1,
    E_BADARGS = -2,
    E_BADALLOC = -3
};

int merge_sort(double* arr, size_t start, size_t end, double* buf);
int merge(double* arr, size_t start, size_t mid, size_t end, double* buf);
int merge_sort_wraper(double* arr, size_t size);
int read_arr_from_file(const char* path, double** arr, size_t* size);
int write_arr_to_file(const char* path, double* arr, size_t size);

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "[main] Bad num input arguments. Try ./out input_array.dat out_data.dat\n");
        exit(EXIT_FAILURE);
    }

    size_t buff_size = 0;
    double* buffer;
    int ret = read_arr_from_file(argv[1], &buffer, &buff_size);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[main] read buffer from file return error\n");
        exit(EXIT_FAILURE);
    }

    int error = MPI_Init(NULL, NULL);
    if (error != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Init errror = %d\n", error);
        return -1;
    }

    double start_time = MPI_Wtime();

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
        ret = merge_sort_wraper(buffer, buff_size);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[main] merge sort of buffer return error\n");
            exit(EXIT_FAILURE);
        }

        ret = write_arr_to_file(argv[2], buffer, buff_size);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[main] write buffer to file return error\n");
            exit(EXIT_FAILURE);
        }

        double end_time = MPI_Wtime();

        printf("%lg\n", end_time - start_time);
    }
    else
    {
        int node_number = 0;
        error = MPI_Comm_rank(MPI_COMM_WORLD, &node_number);
        if (error != MPI_SUCCESS)
        {
            fprintf(stderr, "MPI_Comm_rank error = %d\n", error);
            MPI_Finalize();
            return -1;
        }

        size_t diap  = buff_size / num_nodes;
        size_t start = node_number * diap;
        size_t end = start + diap;
        if (end > buff_size)
            end = buff_size;

        ret = merge_sort_wraper(buffer + start, diap);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[main] %d merge sort of buffer return error\n", node_number);
            exit(EXIT_FAILURE);
        }

        double* merge_buf = NULL;
        if (node_number % 2 == 0)
        {
            errno = 0;
            merge_buf = (double*) malloc((buff_size + 1) * sizeof(*merge_buf));
            if (merge_buf == NULL)
            {
                perror("[main] can't alloc memory\n");
                exit(EXIT_FAILURE);
            }
        }

        int cur_send = 2;
        while (cur_send <= num_nodes)
        {
            //printf("cur_send = %d\n", cur_send);
            if (node_number % cur_send != 0)
            {
                size_t end_send = diap * (cur_send / 2);
                if (end_send + start > buff_size)
                    end_send = buff_size - start;

                //printf("%ld %ld\n", start, start + end_send);

                error = MPI_Send(buffer + start, end_send, MPI_DOUBLE, (node_number / cur_send) * cur_send, 0, MPI_COMM_WORLD); // end check
                if (error != MPI_SUCCESS)
                {
                    fprintf(stderr, "[main] MPI_Send buffer error = %d\n", error);
                    exit(EXIT_FAILURE);
                }

                /*
                printf("%d send node ", node_number);
                for (size_t i = start; i < start + end_send; i++)
                    printf("%lg ", buffer[i]);
                printf("\n");
                */
                break;
            }
            else
            {
                size_t end_recv = diap * (cur_send / 2);
                if (end_recv + start + diap * (cur_send / 2) > buff_size)
                    end_recv = buff_size - start - diap * (cur_send / 2);

                //printf("%ld\n", end_recv);

                error = MPI_Recv(buffer + start + diap * (cur_send / 2), end_recv, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, NULL); // end check
                if (error != MPI_SUCCESS)
                {
                    fprintf(stderr, "MPI_Recv buffer error = %d\n", error);
                    exit(EXIT_FAILURE);
                }
                size_t mid = start + (diap * cur_send / 2 + end_recv) / 2;

                /*
                printf("%d recv node ", node_number);
                for (size_t i = start; i < start + diap * (cur_send / 2) + end_recv; i++)
                    printf("%lg ", buffer[i]);
                printf("\n");
                */
                //printf("%ld %ld %ld\n", start, mid, start + diap * (cur_send / 2) + end_recv);

                merge(buffer, start, mid, start + diap * (cur_send / 2) + end_recv, merge_buf);
                cur_send *= 2;
            }
        }

        if (node_number == 0)
        {
            //ret = write_arr_to_file(argv[2], buffer, buff_size);
            if (ret != E_SUCCESS)
            {
                fprintf(stderr, "[main] write buffer to file return error\n");
                exit(EXIT_FAILURE);
            }

            double end_time = MPI_Wtime();

            printf("%lg\n", end_time - start_time);
        }
    }

    MPI_Finalize();

    //free(buffer);

    return 0;
}

int merge_sort_wraper(double* arr, size_t size)
{
    if (arr == NULL)
    {
        fprintf(stderr, "[merge_sort_wraper] input arr is NULL\n");
        return E_BADARGS;
    }

    //printf("merge_wrapper started\n");

    errno = 0;
    double* buf = (double*) malloc((size + 1) * sizeof(*buf));
    if (buf == NULL)
    {
        perror("[merge_sort_wraper] alloc memory swap buffer error\n");
        return E_BADALLOC;
    }

    int ret = merge_sort(arr, 0, size, buf);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[merge_sort_wraper] merge_sort return error\n");
        return E_ERROR;
    }

    free(buf);

    return E_SUCCESS;
}

int merge(double* arr, size_t start, size_t mid, size_t end, double* buf)
{
    if (arr == NULL || buf == NULL)
    {
        fprintf(stderr, "[merge] input arr is NULL\n");
        return E_BADARGS;
    }

    //printf("%ld %ld %ld\n", start, mid, end);

    for (size_t i = start; i < mid; i++)
        buf[i] = arr[i];

    for (size_t i = mid; i < end; i++)
        buf[i] = arr[i];

    size_t cur_in_left = start;
    size_t cur_in_right = mid;
    size_t cur = start;
    while (cur_in_left < mid && cur_in_right < end)
    {
        if (buf[cur_in_left] <= buf[cur_in_right])
        {
            arr[cur] = buf[cur_in_left];
            cur_in_left++;
        }
        else
        {
            arr[cur] = buf[cur_in_right];
            cur_in_right++;
        }
        cur++;
    }

    while(cur_in_left < mid)
    {
        arr[cur] = buf[cur_in_left];
        cur_in_left++;
        cur++;
    }

    while(cur_in_right < end)
    {
        arr[cur] = buf[cur_in_right];
        cur_in_right++;
        cur++;
    }

    //for (int i = start; i < end; i++)
        //printf("%lg ", arr[i]);
    //printf("\n");

    return E_SUCCESS;
}

int merge_sort(double* arr, size_t start, size_t end, double* buf)
{
    if (arr == NULL || buf == NULL)
    {
        fprintf(stderr, "[merge_sort] input arr is NULL\n");
        return E_BADARGS;
    }

    //printf("%ld %ld %ld %ld\n", start, end, end - start, (end - start) / 2);

    if (end - start <= 1)
    {
        return E_SUCCESS;
    }

    size_t mid = start + (end - start) / 2;
    int ret = merge_sort(arr, start, mid, buf);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[merge_sort] sort first half return error\n");
        return E_ERROR;
    }
    //printf("left finished\n");

    ret = merge_sort(arr, mid, end, buf);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[merge_sort] sort second half return error\n");
        return E_ERROR;
    }
    //printf("right finished\n");

    ret = merge(arr, start, mid, end, buf);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[merge_sort] merge two halfs return error\n");
        return E_ERROR;
    }

    return E_SUCCESS;
}

int write_arr_to_file(const char* path, double* arr, size_t size)
{
    if (path == 0 || arr == NULL)
    {
        fprintf(stderr, "[write_arr_to_file] Bad input arguments\n");
        return E_BADARGS;
    }

    errno = 0;
    FILE* output = fopen(path, "w");
    if (output == NULL)
    {
        perror("[write_arr_to_file] Open file return error\n");
        exit(EXIT_FAILURE);
    }

    fprintf(output, "%ld\n", size);

    for (size_t i = 0; i < size; i++)
        fprintf(output, "%lg ", arr[i]);

    fclose(output);

    return E_SUCCESS;
}

int read_arr_from_file(const char* path, double** arr, size_t* size)
{
    if (path == NULL || arr == NULL || size == NULL)
    {
        fprintf(stderr, "[read_arr_from_file] Bad input args\n");
        return E_BADARGS;
    }

    errno = 0;

    FILE* input = fopen(path, "r");
    if (input == NULL)
    {
        perror("[read_arr_from_file] Can't open input file\n");
        return E_ERROR;
    }

    size_t file_size = 0;
    int ret = fscanf(input, "%ld\n", &file_size);
    if (ret != 1)
    {
        fprintf(stderr, "[read_arr_from_file] Can't read num elements\n");
        return E_ERROR;
    }

    double* buff = (double*) malloc(file_size * sizeof(*buff));
    if (buff == NULL)
    {
        perror("[read_arr_from_file] Bad array allocation\n");
        return E_BADALLOC;
    }

    for (size_t i = 0; i < file_size; i++)
    {
        ret = fscanf(input, "%lg ", &buff[i]);
        if (ret == 0)
        {
            fprintf(stderr, "[read_arr_from_file] Bad double reading\n");
            return E_ERROR;
        }
    }
    fclose(input);

    *arr = buff;
    *size = file_size;

    return E_SUCCESS;
}
