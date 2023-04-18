#include <iostream>

using namespace std;

__global__ void vectorAdd(int *a, int *b, int *c, int vector_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vector_size)
        c[i] = a[i] + b[i];
}

int main(void) {
    cout << "Defining variables & allocating memory in HOST..." << endl;
    int vector_size = 1000000;
    size_t n_bytes = sizeof(int) * vector_size;
    int *a = (int *) malloc(n_bytes);
    int *b = (int *) malloc(n_bytes);
    int *c = (int *) malloc(n_bytes);

    cout << "Defining variables & allocating memory in DEVICE..." << endl;
    int *device_a, *device_b, *device_c;
    cudaMalloc((void **) &device_a, n_bytes);
    cudaMalloc((void **) &device_b, n_bytes);
    cudaMalloc((void **) &device_c, n_bytes);

    cout << "Initializing variables in HOST..." << endl;
    for (int i = 0; i < vector_size; i++) {
        a[i] = i;
        b[i] = vector_size - i;
    }

    cout << "Copying HOST variables to DEVICE variables..." << endl;
    cudaMemcpy(device_a, a, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, n_bytes, cudaMemcpyHostToDevice);

    cout << "Defining & calling kernel..." << endl;
    int block_size = 1024;
    int blocks_count = (vector_size / block_size) + (vector_size % block_size != 0);
    vectorAdd<<<blocks_count, block_size>>>(device_a, device_b, device_c, vector_size);

    cout << "Copying DEVICE variables to HOST variables..." << endl;
    cudaMemcpy(c, device_c, n_bytes, cudaMemcpyDeviceToHost);

    cout << "Checking result..." << endl;
    bool pass = true;
    for (int i = 0; i < vector_size; i++) {
        if (c[i] != a[i] + b[i]) {
            pass = false;
            break;
        }
    }
    if (pass)
        cout << "Passed!" << endl;
    else
        cout << "Failed!" << endl;
    
    cout << "Freeing memory..." << endl;
    free(a);
    free(b);
    free(c);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}
