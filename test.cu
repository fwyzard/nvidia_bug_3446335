#include <cuda_runtime.h>
#include "cuda_check.h"

int main(void) {
  // linear gpu0_buffer layout
  constexpr size_t size = 1024;

  char host_buffer[size];
  memset(host_buffer, 0x00, size);

  // create a CUDA stream on the first device
  cudaStream_t stream0;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaStreamCreate(&stream0));

  // create a CUDA stream on the second device
  cudaStream_t stream1;
  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaStreamCreate(&stream1));

  // ===========================================================================

  // allocate device memory on the first device
  char* gpu0_buffer = nullptr;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc(&gpu0_buffer, size));

  // allocate device memory on the second device
  char* gpu1_buffer = nullptr;
  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaMalloc(&gpu1_buffer, size));

  // cudaMemcpy variants can be called with memory a different device
  CUDA_CHECK(cudaMemcpy(host_buffer, gpu0_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu0_buffer, host_buffer, size, cudaMemcpyDefault));

  CUDA_CHECK(cudaMemcpy(host_buffer, gpu1_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu1_buffer, host_buffer, size, cudaMemcpyDefault));

  CUDA_CHECK(cudaMemcpy(gpu1_buffer, gpu0_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu0_buffer, gpu1_buffer, size, cudaMemcpyDefault));

  // cudaMemcpyAsync variants can be called with memory a different device
  CUDA_CHECK(cudaMemcpyAsync(host_buffer, gpu0_buffer, size, cudaMemcpyDefault, stream0));
  CUDA_CHECK(cudaMemcpyAsync(gpu0_buffer, host_buffer, size, cudaMemcpyDefault, stream0));

  CUDA_CHECK(cudaMemcpyAsync(host_buffer, gpu1_buffer, size, cudaMemcpyDefault, stream0));
  CUDA_CHECK(cudaMemcpyAsync(gpu1_buffer, host_buffer, size, cudaMemcpyDefault, stream0));

  CUDA_CHECK(cudaMemcpyAsync(gpu1_buffer, gpu0_buffer, size, cudaMemcpyDefault, stream0));
  CUDA_CHECK(cudaMemcpyAsync(gpu0_buffer, gpu1_buffer, size, cudaMemcpyDefault, stream0));

  // device memory can be freed with a different current device
  CUDA_CHECK(cudaFree(gpu0_buffer));
  CUDA_CHECK(cudaFree(gpu1_buffer));

  // ===========================================================================

  // allocate stream-ordered device memory on the first device (determined by the stream)
  CUDA_CHECK(cudaMallocAsync(&gpu0_buffer, size, stream0));

  // allocate stream-ordered device memory on the second device (determined by the stream)
  CUDA_CHECK(cudaMallocAsync(&gpu1_buffer, size, stream1));

  // cudaMemcpy variants **cannot** be called with stream-ordered memory on a different device,
  // select the correct device first
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemcpy(host_buffer, gpu0_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu0_buffer, host_buffer, size, cudaMemcpyDefault));

  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaMemcpy(host_buffer, gpu1_buffer, size, cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(gpu1_buffer, host_buffer, size, cudaMemcpyDefault));

  // stream-ordered memory **cannot** be copied across devices with cudaMemcpy,
  // use cudaMemcpyPeer
  CUDA_CHECK(cudaMemcpyPeer(gpu1_buffer, 1, gpu0_buffer, 0, size));
  CUDA_CHECK(cudaMemcpyPeer(gpu0_buffer, 0, gpu1_buffer, 1, size));

  // cudaMemcpyAsync variants **cannot** be called with stream-ordered memory on a different device,
  // select the correct device first
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMemcpyAsync(host_buffer, gpu0_buffer, size, cudaMemcpyDefault, stream0));
  CUDA_CHECK(cudaMemcpyAsync(gpu0_buffer, host_buffer, size, cudaMemcpyDefault, stream0));

  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaMemcpyAsync(host_buffer, gpu1_buffer, size, cudaMemcpyDefault, stream0));
  CUDA_CHECK(cudaMemcpyAsync(gpu1_buffer, host_buffer, size, cudaMemcpyDefault, stream0));

  // stream-ordered memory **cannot** be copied across devices with cudaMemcpyAsync
  // use cudaMemcpyPeerAsync
  CUDA_CHECK(cudaMemcpyPeerAsync(gpu1_buffer, 1, gpu0_buffer, 0, size, stream0));
  CUDA_CHECK(cudaMemcpyPeerAsync(gpu0_buffer, 0, gpu1_buffer, 1, size, stream0));

  // stream-ordered device memory can be freed with cudaFree (without further synchronisation),
  // or with cudaFreeAsync (passing the stream where the operation should be ordered, which can be
  // for a different device than the memory pool)
  CUDA_CHECK(cudaFreeAsync(gpu0_buffer, stream0));
  CUDA_CHECK(cudaFreeAsync(gpu1_buffer, stream0));

  // ===========================================================================

  CUDA_CHECK(cudaStreamDestroy(stream0));
  CUDA_CHECK(cudaStreamDestroy(stream1));
}
