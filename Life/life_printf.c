#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <mpi.h>
#include "SFML/Graphics.hpp"

enum errors{
    E_SUCCESS  = 0,
    E_ERROR    = -1,
    E_BADARGS  = -2,
    E_BADALLOC = -3,
    E_BADMPI   = -4
};

const char initial_path[] = "graph_initial.dat";
const int  NUM_ITERATIONS = 1000;

int read_state(uint8_t** out_buff, size_t* x, size_t* y, const char* file_name);
int entire_evolution(uint8_t* map, uint8_t* new_map, size_t x_len, size_t y_len);
int fix_borders(uint8_t* map, uint8_t* new_map, size_t x_len, size_t y_len, uint8_t* top_border, uint8_t* down_border);

int main(int argc, char* argv[])
{
    size_t x_len, y_len;
    uint8_t* arr = NULL;

    int ret = read_state(&arr, &x_len, &y_len, initial_path);
    if (ret != E_SUCCESS)
    {
        fprintf(stderr, "[main] read state returned error %d\n", ret);
        exit(EXIT_FAILURE);
    }

    sf::RenderWindow window(sf::VideoMode(x_len, y_len), "SFML works!");
    window.setFramerateLimit(60);

    sf::VertexArray draw_buff(sf::Points, x_len * y_len);

    for (int i = 0; i < x_len * y_len; i++)
    {
        draw_buff[i].position = sf::Vector2f(i % x_len, i / y_len);
        draw_buff[i].color = sf::Color::Black;
    }

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
        errno = 0;
        uint8_t* new_arr = (uint8_t*) calloc(x_len * y_len, sizeof(*new_arr));
        if (new_arr == NULL)
        {
            perror("[main] can't alloc next iteration array\n");
            exit(EXIT_FAILURE);
        }

        for (int iter = 0; iter < NUM_ITERATIONS; iter++)
        {
            if (iter % 100 == 0)
                printf("num_iteration %d\n", iter);

            ret = entire_evolution(arr, new_arr, x_len, y_len);
            if (ret != E_SUCCESS)
            {
                fprintf(stderr, "[main] On one node entire_evolution returned error %d\n", ret);
                exit(EXIT_FAILURE);
            }

            uint8_t* top_border = arr + (y_len - 1) * x_len; // swap
            uint8_t* down_border = arr;

            ret = fix_borders(arr, new_arr, x_len, y_len, top_border, down_border);
            if (ret != E_SUCCESS)
            {
                fprintf(stderr, "[main] fix borders in one node mode returned error %d\n", ret);
                exit(EXIT_FAILURE);
            }

            uint8_t* temp = arr;
            arr = new_arr;
            new_arr = temp;

            if (!window.isOpen())
            {
                fprintf(stderr, "[main] Window was closed\n");
                exit(EXIT_FAILURE);
            }

            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed)
                    window.close();
            }

            size_t counter = 0;
            for (size_t i = 0; i < x_len * y_len; i++)
            {
                if (arr[i] == 1)
                {
                    draw_buff[i].color = sf::Color::White;
                    counter++;
                }
                else
                    draw_buff[i].color = sf::Color::Black;
            }

            window.clear();
            window.draw(draw_buff);
            window.display();

            //printf("%ld\n", counter);
        }
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

        size_t y_step = y_len / num_nodes;

        size_t my_y_len = y_step;
        if ((node_number + 1) * y_step > y_len)
            my_y_len = y_len - node_number * y_step;

        size_t y_start = y_step * node_number;

        uint8_t* aligned_buff;
        if (node_number == 0)
        {
            aligned_buff = (uint8_t*) calloc(x_len * num_nodes * y_step, sizeof(*aligned_buff));
            if (aligned_buff == NULL)
            {
                perror("[main] can't alloc aligned buff\n");
                exit(EXIT_FAILURE);
            }
        }

        errno = 0;
        uint8_t* arr_block = (uint8_t*) calloc(x_len * (my_y_len + 2), sizeof(*arr_block));
        if (arr_block == NULL)
        {
            perror("[main] can't alloc zero node arr block array\n");
            exit(EXIT_FAILURE);
        }

        for (size_t i = 0; i < my_y_len; i++)
            for (size_t j = 0; j < x_len; j++)
                arr_block[i * x_len + j] = arr[(y_start + i) * x_len + j];

        errno = 0;
        uint8_t* new_arr = (uint8_t*) calloc(x_len * (my_y_len + 2), sizeof(*new_arr));
        if (new_arr == NULL)
        {
            perror("[main] can't alloc zero node next iteration array\n");
            exit(EXIT_FAILURE);
        }

        for (int iter = 0; iter < NUM_ITERATIONS; iter++)
        {
            if (iter % 100 == 0)
                printf("num_iteration %d\n", iter);

            MPI_Request top_send, down_send;

            if (node_number == 0)
                MPI_Isend(arr_block, x_len, MPI_BYTE, num_nodes - 1, 0, MPI_COMM_WORLD, &top_send);
            else
                MPI_Isend(arr_block, x_len, MPI_BYTE, node_number - 1, 0, MPI_COMM_WORLD, &top_send);

            if (node_number == num_nodes - 1)
                MPI_Isend(arr_block + x_len * (my_y_len - 1), x_len, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &down_send);
            else
                MPI_Isend(arr_block + x_len * (my_y_len - 1), x_len, MPI_BYTE, node_number + 1, 0, MPI_COMM_WORLD, &down_send);

            uint8_t* top_border = arr_block + x_len * my_y_len;
            uint8_t* down_border = arr_block + x_len * (my_y_len + 1);

            if (node_number == 0)
                MPI_Recv(top_border, x_len, MPI_BYTE, num_nodes - 1, 0, MPI_COMM_WORLD, NULL); // top borders
            else
                MPI_Recv(top_border, x_len, MPI_BYTE, node_number - 1, 0, MPI_COMM_WORLD, NULL); // down borders

            if (node_number == num_nodes - 1)
                MPI_Recv(down_border, x_len, MPI_BYTE, 0, 0, MPI_COMM_WORLD, NULL); // down borders
            else
                MPI_Recv(down_border, x_len, MPI_BYTE, node_number + 1, 0, MPI_COMM_WORLD, NULL); // down borders

            MPI_Status status;
            MPI_Wait(&top_send, &status);
            MPI_Wait(&down_send, &status);

            ret = entire_evolution(arr_block, new_arr, x_len, my_y_len);
            if (ret != E_SUCCESS)
            {
                fprintf(stderr, "[main] On one node entire_evolution returned error %d\n", ret);
                exit(EXIT_FAILURE);
            }

            ret = fix_borders(arr_block, new_arr, x_len, my_y_len, top_border, down_border);
            if (ret != E_SUCCESS)
            {
                fprintf(stderr, "[main] fix borders in one node mode returned error %d\n", ret);
                exit(EXIT_FAILURE);
            }

            uint8_t* temp = arr;
            arr = new_arr;
            new_arr = temp;

            int error = MPI_Gather(arr_block, my_y_len * x_len, MPI_BYTE, aligned_buff, y_step * x_len, MPI_BYTE, 0, MPI_COMM_WORLD);
            if (error != MPI_SUCCESS)
            {
                fprintf(stderr, "gather sum error = %d with %d\n", error, node_number);
                MPI_Finalize();
                return -1;
            }

            if (node_number == 0)
            {
                if (!window.isOpen())
                {
                    fprintf(stderr, "[main] Window was closed\n");
                    exit(EXIT_FAILURE);
                }

                sf::Event event;
                while (window.pollEvent(event))
                {
                    if (event.type == sf::Event::Closed)
                        window.close();
                }

                size_t counter = 0;
                for (size_t i = 0; i < x_len * y_len; i++)
                {
                    if (aligned_buff[i] == 1)
                    {
                        draw_buff[i].color = sf::Color::White;
                        counter++;
                    }
                    else
                        draw_buff[i].color = sf::Color::Black;
                }

                window.clear();
                window.draw(draw_buff);
                window.display();
            }

        }
    }

    MPI_Finalize();
    return 0;
}

int read_state(uint8_t** out_buff, size_t* x, size_t* y, const char* file_name)
{
    if (file_name == NULL || x == NULL || y == NULL)
    {
        fprintf(stderr, "[read_state]Bad input ptr\n");
        return E_BADARGS;
    }

    errno = 0;
    FILE* input = fopen(file_name, "rb");
    if (input == NULL)
    {
        perror("[read_state] Bad opening file\n");
        return E_ERROR;
    }

    size_t x_len = 0;
    size_t y_len = 0;
    int ret = fscanf(input,"%ld\n%ld\n", &x_len, &y_len);
    if (ret != 2)
    {
        fprintf(stderr, "[read_state] Can't scan x size  = %ld and y size != %ld\n", x_len, y_len);
        return E_ERROR;
    }

    uint8_t* map = (uint8_t*)calloc(x_len * y_len, sizeof(*map));
    if (map == NULL)
    {
        perror("[read_state] Alloc memory block for map returned error\n");
        return E_BADALLOC;
    }

    for (size_t y_coord = 0; y_coord < y_len; y_coord++)
    {
        for (size_t x_coord = 0; x_coord < x_len; x_coord++)
            fscanf(input, "%hhd ", &map[y_coord * x_len + x_coord]);
        fscanf(input, "\n");
    }

    *out_buff = map;
    *x = x_len;
    *y = y_len;

    return E_SUCCESS;
}

int entire_evolution(uint8_t* map, uint8_t* new_map, size_t x_len, size_t y_len)
{
    if (map == NULL || new_map == NULL)
    {
        fprintf(stderr, "[entire_evolution] Bad input maps\n");
        return E_BADARGS;
    }

    for (size_t y = 1; y < y_len - 1; y++)
    {
        size_t y_prev = y - 1;
        size_t y_next = y + 1;
        for (size_t x = 0; x < x_len; x++)
        {
            size_t x_prev = x - 1;
            if (x == 0)
                x_prev = x_len - 1;

            size_t x_next = x + 1;
            if (x == x_len - 1)
                x_next = 0;

            uint8_t sum_prev = map[y_prev * x_len + x_prev] + map[y_prev * x_len + x] + map[y_prev * x_len + x_next];
            uint8_t sum_cur  = map[y * x_len + x_prev] + map[y * x_len + x_next];
            uint8_t sum_next = map[y_next * x_len + x_prev] + map[y_next * x_len + x] + map[y_next * x_len + x_next];
            uint8_t sum = sum_prev + sum_cur + sum_next;

            if (map[y * x_len + x] == 0)
            {
                if (sum == 3)
                    new_map[y * x_len + x] = 1;
                else
                    new_map[y * x_len + x] = 0;
            }
            else
            {
                //printf("I'm white\n");
                if (sum != 2 && sum != 3)
                    new_map[y * x_len + x] = 0;
                else
                    new_map[y * x_len + x] = 1;
            }
        }
    }

    return E_SUCCESS;
}

int fix_borders(uint8_t* map, uint8_t* new_map, size_t x_len, size_t y_len, uint8_t* top_border, uint8_t* down_border)
{
    if (map == NULL || new_map == NULL || top_border == NULL || down_border == NULL)
    {
        fprintf(stderr, "[fix_borders] Bad input arrays\n");
        return E_BADARGS;
    }

    for (size_t x = 0; x < x_len; x++)
    {
        size_t x_prev = x - 1;
        if (x == 0)
            x_prev = x_len - 1;

        size_t x_next = x + 1;
        if (x == x_len - 1)
            x_next = 0;

        size_t y_next_top = 1;
        size_t y_top      = 0;
        size_t y_prev_down = y_len - 2;
        size_t y_down      = y_len - 1;

        uint8_t sum_prev = top_border[x_prev] + top_border[x] + top_border[x_next];
        uint8_t sum_cur  = map[y_top * x_len + x_prev] + map[y_top * x_len + x_next];
        uint8_t sum_next = map[y_next_top * x_len + x_prev] + map[y_next_top * x_len + x] + map[y_next_top * x_len + x_next];
        uint8_t sum_top  = sum_prev + sum_cur + sum_next;

        sum_prev = map[y_prev_down * x_len + x_prev] + map[y_prev_down * x_len + x] + map[y_prev_down * x_len + x_next];
        sum_cur  = map[y_down * x_len + x_prev] + map[y_down * x_len + x_next];
        sum_next = down_border[x_prev] + down_border[x] + down_border[x_next];
        uint8_t sum_down = sum_prev + sum_cur + sum_next;

        if (map[y_top * x_len + x] == 0)
        {
            if (sum_top == 3)
                new_map[y_top * x_len + x] = 1;
        }
        else
        {
            if (sum_top != 2 && sum_top != 3)
                new_map[y_top * x_len + x] = 0;
        }

        if (map[y_down * x_len + x] == 0)
        {
            if (sum_down == 3)
                new_map[y_down * x_len + x] = 1;
        }
        else
        {
            if (sum_down != 2 && sum_down != 3)
                new_map[y_down * x_len + x] = 0;
        }
    }

    return E_SUCCESS;
}
