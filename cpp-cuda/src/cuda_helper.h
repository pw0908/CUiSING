#ifndef __CUDA_MACRO_H__
#define __CUDA_MACRO_H__

#define CHECK_CUDA(call) {                                                   \
    cudaError_t err = call;                                                  \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_CUBLAS(call) {                                                 \
    cublasStatus_t status = call;                                            \
    if( CUBLAS_STATUS_SUCCESS != status) {                                   \
        fprintf(stderr, "CUBLAS error: %s = %d at (%s:%d)\n", #call,         \
                status, __FILE__, __LINE__);                                 \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_CURAND(call) {                                                 \
    curandStatus_t status = call;                                            \
    if( CURAND_STATUS_SUCCESS != status) {                                   \
        fprintf(stderr, "CURAND error: %s = %d at (%s:%d)\n", #call,         \
                status, __FILE__, __LINE__);                                 \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_ERROR(errorMessage) {                                          \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }}
#endif