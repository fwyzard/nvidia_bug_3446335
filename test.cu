#include <cuda_runtime.h>
#include "cuda_check.h"

int main(void) {
  // linear gpu0_buffer layout
  constexpr size_t size = 1024;

  char host_buffer[size];
  memset(host_buffer, 0x00, size);

  // select the first device
  CUDA_CHECK(cudaSetDevice(0));

  // create a CUDA stream on the first device
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // ===========================================================================

  // allocate device memory on the first device
  char* gpu0_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&gpu0_buffer, size));

  // select the second device
  CUDA_CHECK(cudaSetDevice(1));

  // allocate device memory on the second device
  char* gpu1_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&gpu1_buffer, size));

  // cudaMemcpy variants can be called with memory a different device
  CUDA_CHECK(cudaMemcpy(host_buffer, gpu0_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu0_buffer, host_buffer, size, cudaMemcpyDefault));

  CUDA_CHECK(cudaMemcpy(host_buffer, gpu1_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu1_buffer, host_buffer, size, cudaMemcpyDefault));

  CUDA_CHECK(cudaMemcpy(gpu1_buffer, gpu0_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu0_buffer, gpu1_buffer, size, cudaMemcpyDefault));

  // cudaMemcpyAsync variants can be called with memory a different device
  CUDA_CHECK(cudaMemcpyAsync(host_buffer, gpu0_buffer, size, cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaMemcpyAsync(gpu0_buffer, host_buffer, size, cudaMemcpyDefault, stream));

  CUDA_CHECK(cudaMemcpyAsync(host_buffer, gpu1_buffer, size, cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaMemcpyAsync(gpu1_buffer, host_buffer, size, cudaMemcpyDefault, stream));

  CUDA_CHECK(cudaMemcpyAsync(gpu1_buffer, gpu0_buffer, size, cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaMemcpyAsync(gpu0_buffer, gpu1_buffer, size, cudaMemcpyDefault, stream));

  // free the device memory
  CUDA_CHECK(cudaFree(gpu0_buffer));

  // ===========================================================================

  // select the first device
  CUDA_CHECK(cudaSetDevice(0));

  // allocate stream-ordered device memory on the first device
  CUDA_CHECK(cudaMallocAsync(&gpu0_buffer, size, stream));

  // select the second device
  CUDA_CHECK(cudaSetDevice(1));

  // cudaMemcpy variants **cannot** be called with stream-ordered memory on a different device
  CUDA_CHECK(cudaMemcpy(host_buffer, gpu0_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu0_buffer, host_buffer, size, cudaMemcpyDefault));

  CUDA_CHECK(cudaMemcpy(host_buffer, gpu1_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu1_buffer, host_buffer, size, cudaMemcpyDefault));

  // stream-ordered memory **cannot** be copied across devices
  CUDA_CHECK(cudaMemcpy(gpu1_buffer, gpu0_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu0_buffer, gpu1_buffer, size, cudaMemcpyDefault));

  // cudaMemcpyAsync variants **cannot** be called with stream-ordered memory on a different device
  CUDA_CHECK(cudaMemcpyAsync(host_buffer, gpu0_buffer, size, cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaMemcpyAsync(gpu0_buffer, host_buffer, size, cudaMemcpyDefault, stream));

  CUDA_CHECK(cudaMemcpyAsync(host_buffer, gpu1_buffer, size, cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaMemcpyAsync(gpu1_buffer, host_buffer, size, cudaMemcpyDefault, stream));

  // stream-ordered memory **cannot** be copied across devices
  CUDA_CHECK(cudaMemcpyAsync(gpu1_buffer, gpu0_buffer, size, cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaMemcpyAsync(gpu0_buffer, gpu1_buffer, size, cudaMemcpyDefault, stream));

  // free the stream-ordered device memory
  CUDA_CHECK(cudaFreeAsync(gpu0_buffer, stream));

  // ===========================================================================

  CUDA_CHECK(cudaStreamDestroy(stream));
}
