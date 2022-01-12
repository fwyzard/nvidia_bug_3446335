# nvidia_bug_3446335
A simple program to reproduce NVIDIA bug 3446335

## building and running
```bash
git clone https://github.com/fwyzard/nvidia_bug_3446335.git
cd nvidia_bug_3446335
make
./test
```

## rationale

The example shows that calling `cudaMemcpy()` or `cudaMemcpyAsync()` for copying to or from a buffer allocated by `cudaMallocAsync()` fails if the current device is not the one that holds the memory.
Memory allocated by `cudaMalloc()` does not have this restriction.

For copies to/from the host, the solution is to set the correct device.

For copies between different devices, the solution is to use `cudaMemcpyPeer()` or `cudaMemcpyPeerAsync()`.
