#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>

enum errors
{
    E_SUCCESS  = 0,
    E_ERROR    = -1,
    E_BADARGS  = -2,
    E_BADALLOC = -3,
    E_FLOAT    = -4,
};

const double START = 0.001;
const double END   = 1000.0;
const double DOUBLE_PREC = 0.00000000001;
const size_t SCALE_PARAM = 10;

typedef struct task
{
    double start_diap;
    double end_diap;
    double result;
    double eps;
} task;

typedef struct thread_info
{
    task* calc_tasks;
    pthread_mutex_t* mutex;
    size_t* next;
    size_t* finished;
    size_t num_tasks;
} thread_info;

int func(double* res, double x);
int integral(double* ans, double start, double end, double start_val, double end_val, double prec);
int cmp_double(double a, double b);
void* thread_routine(void* data);

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("Bad number input arguments. Try ./integr.out [num_thread] [precision]\n");
        return 0;
    }

    errno = 0;
    long int num_threads = strtol(argv[1], NULL, 10);
    if (errno < 0)
    {
        perror("[main] Getting num threads returned error\n");
        exit(EXIT_FAILURE);
    }

    errno = 0;
    double prec = strtod(argv[2], NULL);
    if (errno < 0)
    {
        perror("[main] Getting precision returned error\n");
        exit(EXIT_FAILURE);
    }

    size_t num_tasks = num_threads * SCALE_PARAM;
    errno = 0;
    task* task_arr = (task*) malloc(sizeof(*task_arr) * num_tasks);
    if (task_arr == NULL)
    {
        perror("[main] alloc task array returned error\n");
        exit(EXIT_FAILURE);
    }

    double step = (END - START) / num_tasks;
    for (size_t i = 0; i < num_tasks; i++)
    {
        task_arr[i].start_diap = START + i * step;
        task_arr[i].end_diap = task_arr[i].start_diap + step;
        task_arr[i].result = 0;
        task_arr[i].eps = prec;
    }

    size_t next_task = 0;
    size_t last_finished = 0;

    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    errno = 0;
    thread_info* thread_info_arr = (thread_info*) malloc(sizeof(*thread_info_arr) * num_threads);
    if (thread_info_arr == NULL)
    {
        perror("[main] alloc task array returned error\n");
        exit(EXIT_FAILURE);
    }

    errno = 0;
    pthread_t* thread_arr = (pthread_t*) malloc(sizeof(*thread_arr) * num_threads);
    if (thread_arr == NULL)
    {
        perror("[main] can't alloc thread array\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < num_threads; i++)
    {
        thread_info_arr[i].mutex = &mutex;
        thread_info_arr[i].calc_tasks = task_arr;
        thread_info_arr[i].next = &next_task;
        thread_info_arr[i].finished = &last_finished;
        thread_info_arr[i].num_tasks = num_tasks;

        errno = 0;
        int ret = pthread_create(&thread_arr[i], NULL, thread_routine, &thread_info_arr[i]);
        if (ret < 0)
        {
            perror("[main] can't start thread routine\n");
            exit(EXIT_FAILURE);
        }
    }

    for (size_t i = 0; i < num_threads; i++)
    {
        pthread_join(thread_arr[i], NULL);
    }

    double answer = 0.0;
    for (size_t i = 0; i < num_tasks; i++)
        answer += task_arr[i].result;

    printf("answer = %lg\n", answer);

    return 0;
}

int cmp_double(double a, double b)
{
    double diff = a - b;
    if (diff < 0)
    {
        if (diff >= -DOUBLE_PREC)
            return 0;
        else
            return -1;
    }
    else
    {
        if (diff <= DOUBLE_PREC)
            return 0;
        else
            return 1;
    }
}

int func(double* res, double x)
{
    if (x != x) // NaN check
    {
        fprintf(stderr, "[func] input x is NaN");
        return E_BADARGS;
    }

    if (res == NULL)
    {
        fprintf(stderr, "[func] output res pointer is NULL");
        return E_BADARGS;
    }

    if (x == 0)
        return E_FLOAT;

    *res = sin(1 / x);

    return E_SUCCESS;
}

void* thread_routine(void* data)
{
    if (data == NULL)
    {
        fprintf(stderr, "[thread_routine] input data is NULL\n");
        exit(EXIT_FAILURE);
    }

    thread_info* info = (thread_info*) data;
    task* task_arr = info->calc_tasks;
    pthread_mutex_t* mutex = info->mutex;
    size_t* next_task = info->next;
    size_t  max_task_num = info->num_tasks;

    size_t curr_task = 0;
    while(*next_task < max_task_num)
    {
        pthread_mutex_lock(mutex);
        curr_task = *next_task;
        if (curr_task >= max_task_num)
        {
            pthread_mutex_unlock(mutex);
            break;
        }
        (*next_task)++;
        pthread_mutex_unlock(mutex);

        double start = task_arr[curr_task].start_diap;
        double end   = task_arr[curr_task].end_diap;

        double start_val = 0.0;
        double end_val   = 0.0;
        func(&start_val, start);
        func(&end_val, end);

        double result = 0.0;
        int ret = integral(&result, start, end, start_val, end_val, task_arr[curr_task].eps);
        if (ret != E_SUCCESS)
        {
            fprintf(stderr, "[thread_routine] integral returned error - %d\n", ret);
            exit(EXIT_FAILURE);
        }

        task_arr[curr_task].result = result;
    }

    return 0;
}

int integral(double* ans, double start, double end, double start_val, double end_val, double prec)
{
    static size_t num_calls = 0;

    if (ans == NULL)
    {
        fprintf(stderr, "[integral] output ans pointer is NULL\n");
        return E_BADARGS;
    }

    double mid = (start + end) / 2;
    double mid_val = 0.0;
    int ret = func(&mid_val, mid);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[integral] calc mid value returned error - %d\n", ret);
        return E_ERROR;
    }

    double integ = (start_val + end_val) / 2 * (end - start);

    double left_integ  = (start_val + mid_val) / 2 * (mid - start);
    double right_integ = (mid_val + end_val) / 2 * (end - mid);
    double prec_integ  = left_integ + right_integ;

    double diff = prec_integ - integ;
    if (diff < 0)
        diff = -diff;

    double check_val = prec * integ;
    if (check_val < 0)
        check_val = -check_val;

    //printf("num_cals = %ld, diff = %lg, prec * integ = %lg\n", num_calls, diff, prec * integ);

    if (cmp_double(diff, check_val) <= 0)
    {
        *ans = integ;
        return E_SUCCESS;
    }

    ret = integral(&left_integ, start, mid, start_val, mid_val, prec);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[integral] calc precisely left integral returned error %d\n", ret);
        return E_ERROR;
    }

    ret = integral(&right_integ, mid, end, mid_val, end_val, prec);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[integral] calc precisely right integral returned error %d\n", ret);
        return E_ERROR;
    }

    *ans = left_integ + right_integ;

    return E_SUCCESS;
}
