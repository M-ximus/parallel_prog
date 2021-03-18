#include <stdio.h>
#include <gmp.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

const unsigned long int prec = 1000;
const int N = 1000;

int main(int argc, char* argv[])
{
    unsigned long int power = (unsigned long int)(((double)prec) * 3.3219);
    mpf_set_default_prec(power);

    mpf_t eps;
    mpf_init_set_ui(eps, 10);
    mpf_pow_ui(eps, eps, power);
    mpf_ui_div(eps, 1, eps);

    //printf("%ld\n", power);
    //gmp_printf("%.*Ff\n", power, eps);

    mpf_clear(eps);

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

////////////////////////////////////////////////////////////////////////////////
    if (num_nodes == 1)
    {
        mpf_t sum;
        mpf_init_set_ui(sum, 1);
        mpf_t frac;
        mpf_init_set_ui(frac, 1);
        mpf_t temp;
        mpf_init_set_ui(temp, 1);

        for (int i = 1; i < N; i++)
        {
            mpf_set_ui(frac, i);
            mpf_ui_div(frac, 1, frac);
            mpf_mul(temp, temp, frac);
            mpf_add(sum, sum, temp);
        }

        gmp_printf("%.*Ff\n", prec, sum);
        mpf_clear(sum);
        mpf_clear(frac);
        MPI_Finalize();
        return 0;
    }
////////////////////////////////////////////////////////////////////////////////

    int node_number = 0;
    error = MPI_Comm_rank(MPI_COMM_WORLD, &node_number);
    if (error != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Comm_rank error = %d\n", error);
        MPI_Finalize();
        return -1;
    }

    int diap  = N / num_nodes;
    int start = 1 + node_number * diap;
    int end   = start + diap;
    if (end > N)
        end = N;

    mpf_t correction;
    mpf_init_set_ui(correction, 1);

    if (node_number != 0)
    {
        int i        = start - diap;
        int prev_end = end - diap;
        mpf_t temp;
        mpf_init_set_ui(temp, 1);

        for(; i < prev_end; i++)
        {
            mpf_set_ui(temp, i);
            mpf_ui_div(temp, 1, temp);
            mpf_mul(correction, correction, temp);
        }
        mpf_clear(temp);
    }

    mpf_t part_sum;
    mpf_init_set_ui(part_sum, 0);
    mpf_t frac;
    mpf_init_set_ui(frac, 1);
    mpf_t temp;
    mpf_init_set_ui(temp, 1);

    for (int i = start; i < end; i++)
    {
        mpf_set_ui(frac, i);
        mpf_ui_div(frac, 1, frac);
        mpf_mul(temp, temp, frac);
        mpf_add(part_sum, part_sum, temp);
    }

    mpf_clear(temp);
    mpf_clear(frac);

    mpf_mul(part_sum, part_sum, correction);
    mp_exp_t exp_sum, exp_cor;
    char* part_sum_str   =  mpf_get_str(NULL, &exp_sum, 10, 0, part_sum);
    char* correction_str =  mpf_get_str(NULL, &exp_cor, 10, 0, correction);

    errno = 0;
    size_t buf_len = prec + 1 + 2 + 21 + 1;
    char* buff_sum = (char*) malloc(sizeof(*buff_sum) * buf_len);
    if (buff_sum == NULL)
    {
        perror("Bad allocating sum\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    char* buff_cor = (char*) malloc(sizeof(*buff_cor) * buf_len);
    if (buff_cor == NULL)
    {
        perror("Bad allocating cor\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    exp_sum = exp_sum - strlen(part_sum_str);
    exp_cor = exp_cor - strlen(correction_str);
    sprintf(buff_sum, "%se%ld", part_sum_str, exp_sum);
    sprintf(buff_cor, "%se%ld", correction_str, exp_cor);

    char* recv_buff_sum;
    char* recv_buff_cor;
    if (node_number == 0)
    {
        size_t recv_buff_len = buf_len * num_nodes;
        recv_buff_sum = (char*) malloc(sizeof(*recv_buff_sum) * recv_buff_len);
        if (recv_buff_sum == NULL)
        {
            perror("Bad allocating recv sum buff\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        recv_buff_cor = (char*) malloc(sizeof(*recv_buff_cor) * recv_buff_len);
        if (recv_buff_cor == NULL)
        {
            perror("Bad allocating recv cor buff\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    error = MPI_Gather(buff_sum, buf_len, MPI_CHAR, recv_buff_sum, buf_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (error != MPI_SUCCESS)
    {
        fprintf(stderr, "gather sum error = %d with %d\n", error, node_number);
        MPI_Finalize();
        return -1;
    }

    error = MPI_Gather(buff_cor, buf_len, MPI_CHAR, recv_buff_cor, buf_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (error != MPI_SUCCESS)
    {
        fprintf(stderr, "gather cor error = %d with %d\n", error, node_number);
        MPI_Finalize();
        return -1;
    }

    if (node_number == 0)
    {

        errno = 0;
        char* str = (char*) malloc(sizeof(*str) * buf_len);
        if (str == NULL)
        {
            perror("Bad final allocating error\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        mpf_t recv_sum, sum, part_factor, recv_factor;
        mpf_init(recv_sum);
        mpf_init_set_ui(sum, 1);
        mpf_init_set_ui(part_factor, 1);
        mpf_init(recv_factor);

        for (int i = 0; i < num_nodes; i++)
        {
            strncpy(str, recv_buff_sum + i * buf_len, buf_len);
            mpf_set_str(recv_sum, str, 10);
            strncpy(str, recv_buff_cor + i * buf_len, buf_len);
            mpf_set_str(recv_factor, str, 10);

            mpf_mul(recv_sum, recv_sum, part_factor);
            mpf_mul(part_factor, part_factor, recv_factor);
            mpf_add(sum, sum, recv_sum);
        }

        gmp_printf("%.*Ff\n", prec, sum);
    }

    //gmp_printf("%.*Ff\n", prec, recv_sum);

    MPI_Finalize();

    return 0;
}
