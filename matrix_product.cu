#include <iostream>

using namespace std;

__global__ void matrixProduct(float *matrix_1, float *matrix_2, float *result_matrix, int matrix_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < matrix_size * matrix_size) {
        int row = i / matrix_size;
        int col = i % matrix_size;
        *(result_matrix + matrix_size * row + col) = 0.0;
        for (int k = 0; k < matrix_size; k++) {
            *(result_matrix + matrix_size * row + col) += *(matrix_1 + matrix_size * row + k) * *(matrix_2 + matrix_size * k + col);
        }
    }
}

int main(void) {
    cout << "Defining variables & allocating memory in HOST..." << endl;
    size_t matrix_size = 10240;
    size_t matrix_entries = matrix_size * matrix_size;
    size_t n_bytes = sizeof(float) * matrix_entries;
    float *matrix_1 = (float *) malloc(n_bytes);
    float *matrix_2 = (float *) malloc(n_bytes);
    float *result_matrix = (float *) malloc(n_bytes);

    cout << "Defining variables & allocating memory in DEVICE..." << endl;
    float *device_matrix_1;
    float *device_matrix_2;
    float *device_result_matrix;
    cudaMalloc((void **) &device_matrix_1, n_bytes);
    cudaMalloc((void **) &device_matrix_2, n_bytes);
    cudaMalloc((void **) &device_result_matrix, n_bytes);

    cout << "Initializing variables in HOST..." << endl;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            *(matrix_1 + matrix_size * i + j) = (float) i / (j + 1);
            *(matrix_2 + matrix_size * i + j) = (float) (i + 5) / (j + 8);
        }
    }

    cout << "Copying HOST variables to DEVICE variables..." << endl;
    cudaMemcpy(device_matrix_1, matrix_1, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_2, matrix_2, n_bytes, cudaMemcpyHostToDevice);

    cout << "Defining & calling kernel..." << endl;
    int block_size = 1024;
    int blocks_count = (matrix_entries / block_size) + (matrix_entries % block_size != 0);
    matrixProduct<<<blocks_count, block_size>>>(device_matrix_1, device_matrix_2, device_result_matrix, matrix_size);

    cout << "Copying DEVICE variables to HOST variables..." << endl;
    cudaMemcpy(result_matrix, device_result_matrix, n_bytes, cudaMemcpyDeviceToHost);

    cout << "Checking result..." << endl;
    bool pass = true;
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            float product = 0.0;
            for (int k = 0; k < matrix_size; k++) {
                product += *(matrix_1 + matrix_size * i + k) * *(matrix_2 + matrix_size * k + j);
            }
            if (abs(*(result_matrix + matrix_size * i + j) - product) > 0.1) {
                cout << "result_matrix[" << i << "][" << j << "] = " << *(result_matrix + matrix_size * i + j) << " != " << product << endl;
                pass = false;
                break;
            }
        }
        if (!pass)
            break;
    }
    if (pass)
        cout << "Passed!" << endl;
    else
        cout << "Failed!" << endl;

    cout << "Freeing resources..." << endl;
    free(matrix_1);
    free(matrix_2);
    free(result_matrix);
    cudaFree(device_matrix_1);
    cudaFree(device_matrix_2);
    cudaFree(device_result_matrix);

    return 0;
}
