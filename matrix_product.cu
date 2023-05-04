#include <iostream>
#include <cstdlib>
#include "cublas.h"

#if defined(unix)
#include <sys/time.h>
#elif defined(_WIN32)
#include <time.h>
#endif

using namespace std;

bool has_error = false;
int exit_code = EXIT_SUCCESS;
bool pass;

size_t matrix_size = 1024;
size_t matrix_entries = matrix_size * matrix_size;
size_t n_bytes = sizeof(float) * matrix_entries;
float *matrix_1;
float *matrix_2;
float *result_matrix;
float *device_matrix_1;
float *device_matrix_2;
float *device_result_matrix;

#if defined(unix)
struct timeval cpu_start, cpu_end;
#elif defined(_WIN32)
clock_t cpu_start, cpu_end;
#endif
cudaEvent_t gpu_start, gpu_end;

void allocate_host_memory()
{
    cout << "Allocating memory in HOST..." << endl;
    matrix_1 = (float *)malloc(n_bytes);
    matrix_2 = (float *)malloc(n_bytes);
    result_matrix = (float *)malloc(n_bytes);
    if (!matrix_1 || !matrix_2 || !result_matrix)
    {
        cerr << "Failed to allocate memory in HOST!" << endl;
        exit_code = EXIT_FAILURE;
    }
}

void allocate_device_memory()
{
    cout << "Allocating memory in DEVICE..." << endl;
    has_error |= cublasAlloc(matrix_entries, sizeof(float), (void **)&device_matrix_1) != CUBLAS_STATUS_SUCCESS;
    has_error |= cublasAlloc(matrix_entries, sizeof(float), (void **)&device_matrix_2) != CUBLAS_STATUS_SUCCESS;
    has_error |= cublasAlloc(matrix_entries, sizeof(float), (void **)&device_result_matrix) != CUBLAS_STATUS_SUCCESS;
    if (has_error)
    {
        cerr << "Failed to allocate memory in HOST!" << endl;
        exit_code = EXIT_FAILURE;
    }
}

void initialize_host_variables()
{
    cout << "Initializing variables in HOST..." << endl;
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            *(matrix_1 + matrix_size * i + j) = (float)i / (j + 1);
            *(matrix_2 + matrix_size * i + j) = (float)(i + 5) / (j + 8);
        }
    }
}

void copy_host_to_device()
{
    cout << "Copying HOST variables to DEVICE variables..." << endl;
    has_error |= cublasSetMatrix(matrix_size, matrix_size, sizeof(float), matrix_1, matrix_size, device_matrix_1, matrix_size) != CUBLAS_STATUS_SUCCESS;
    has_error |= cublasSetMatrix(matrix_size, matrix_size, sizeof(float), matrix_2, matrix_size, device_matrix_2, matrix_size) != CUBLAS_STATUS_SUCCESS;
    if (has_error)
    {
        cerr << "Failed to copy HOST variables to DEVICE variables!" << endl;
        exit_code = EXIT_FAILURE;
    }
}

void define_time_variables()
{
    cout << "Defining events and timevals to calculate GPU & CPU time..." << endl;
    has_error |= cudaEventCreate(&gpu_start) != cudaSuccess;
    has_error |= cudaEventCreate(&gpu_end) != cudaSuccess;
    if (has_error)
    {
        cerr << "Failed to create cuda events!" << endl;
        exit_code = EXIT_FAILURE;
    }
}

void calculate_multiplication()
{
    cout << "Calculating multiplication..." << endl;
    has_error |= cudaEventRecord(gpu_start, 0) != cudaSuccess;
    cublasSgemm('n', 'n', matrix_size, matrix_size, matrix_size, 1, device_matrix_1, matrix_size, device_matrix_2, matrix_size, 0, device_result_matrix, matrix_size);
    has_error |= cublasGetError() != CUBLAS_STATUS_SUCCESS;
    has_error |= cudaEventRecord(gpu_end, 0) != cudaSuccess;
    has_error |= cudaEventSynchronize(gpu_end) != cudaSuccess;
    if (has_error)
    {
        cerr << "Failed to calculate multiplication!" << endl;
        exit_code = EXIT_FAILURE;
    }
}

void copy_device_to_host()
{
    cout << "Copying DEVICE variables to HOST variables..." << endl;
    has_error |= cublasGetMatrix(matrix_size, matrix_size, sizeof(float), device_result_matrix, matrix_size, result_matrix, matrix_size) != CUBLAS_STATUS_SUCCESS;
    if (has_error)
    {
        cerr << "Failed to copy DEVICE variables to HOST variables!" << endl;
        exit_code = EXIT_FAILURE;
    }
}

void check_results()
{
    cout << "Checking result..." << endl;
    bool pass = true;
#if defined(unix)
    gettimeofday(&cpu_start, NULL);
#elif defined(_WIN32)
    cpu_start = clock();
#endif
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            float product = 0.0;
            for (int k = 0; k < matrix_size; k++)
            {
                product += *(matrix_1 + matrix_size * k + j) * *(matrix_2 + matrix_size * i + k);
            }
            if (abs(*(result_matrix + matrix_size * i + j) - product) > 1)
            {
                cout << "result_matrix[" << i << "][" << j << "] = " << *(result_matrix + matrix_size * i + j) << " != " << product << endl;
                pass = false;
                exit_code = EXIT_FAILURE;
                break;
            }
        }
        if (!pass)
            break;
    }
#if defined(unix)
    gettimeofday(&cpu_end, NULL);
#elif defined(_WIN32)
    cpu_end = clock();
#endif
    if (pass)
        cout << "Passed!" << endl;
    else
        cout << "Failed!" << endl;
}

void calculate_gpu_cpu_time()
{
    cout << "Calculating GPU & CPU time..." << endl;
    float gpu_elapsed_time;
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_end);
#if defined(unix)
    long seconds = cpu_end.tv_sec - cpu_start.tv_sec;
    long useconds = cpu_end.tv_usec - cpu_start.tv_usec;
    double cpu_elapsed_time = ((seconds)*1000 + useconds / 1000.0);
#elif defined(_WIN32)
    double cpu_elapsed_time = ((double)(cpu_end - cpu_start) / (double)CLOCKS_PER_SEC * 1000);
#endif
    cout << "GPU time: " << gpu_elapsed_time << "ms" << endl;
    cout << "CPU time: " << cpu_elapsed_time << "ms" << endl;
}

void free_resources()
{
    cout << "Shutting down cuBLAS & freeing resources..." << endl;
    has_error = false;
    has_error |= cublasShutdown() != CUBLAS_STATUS_SUCCESS;
    has_error |= cudaEventDestroy(gpu_start) != CUBLAS_STATUS_SUCCESS;
    has_error |= cudaEventDestroy(gpu_end) != CUBLAS_STATUS_SUCCESS;
    free(matrix_1);
    free(matrix_2);
    free(result_matrix);
    has_error |= cublasFree(device_matrix_1) != CUBLAS_STATUS_SUCCESS;
    has_error |= cublasFree(device_matrix_2) != CUBLAS_STATUS_SUCCESS;
    has_error |= cublasFree(device_result_matrix) != CUBLAS_STATUS_SUCCESS;
    if (has_error)
    {
        cerr << "Failed to free resources!" << endl;
        exit_code = EXIT_FAILURE;
    }
}

int main(void)
{
    cout << "Initializing cuBLAS..." << endl;
    cublasInit();

    allocate_host_memory();

    if (exit_code != EXIT_FAILURE)
        allocate_device_memory();

    if (exit_code != EXIT_FAILURE)
        initialize_host_variables();

    if (exit_code != EXIT_FAILURE)
        copy_host_to_device();

    if (exit_code != EXIT_FAILURE)
        define_time_variables();
    
    if (exit_code != EXIT_FAILURE)
        calculate_multiplication();

    if (exit_code != EXIT_FAILURE)
        copy_device_to_host();

    if (exit_code != EXIT_FAILURE)
    {
        check_results();
        calculate_gpu_cpu_time();
    }

    free_resources();

    return exit_code;
}
