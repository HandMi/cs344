/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <cfloat>
#include <stdio.h>

// See
// https://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda

__device__ __forceinline__ void atomicMaxf(float* address, float value) {
  while (value > *address) {
    int old = __float_as_int(*address);
    old = atomicCAS((int*)address, old, __float_as_int(value));
  }
  return;
}

__device__ __forceinline__ void atomicMinf(float* address, float value) {
  while (value < *address) {
    int old = __float_as_int(*address);
    old = atomicCAS((int*)address, old, __float_as_int(value));
  }
  return;
}

__global__ void shmem_getExtrema(const float* const d_array, float* d_min,
                                 float* d_max, uint size) {
  extern __shared__ float sdata[];

  float* min_data = sdata;
  float* max_data = sdata + blockDim.x;

  uint tid = threadIdx.x;
  int id = (blockDim.x * blockIdx.x) + tid;

  if (id < size) {
    min_data[tid] = d_array[id];
    max_data[tid] = d_array[id];
  } else {
    min_data[tid] = FLT_MAX;
    max_data[tid] = -FLT_MAX;
  }

  // Ensure all threads have finished writing
  __syncthreads();

  for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && id < size) {
      min_data[tid] = min(min_data[tid], min_data[tid + s]);
      max_data[tid] = max(max_data[tid], max_data[tid + s]);
    }
    __syncthreads();
  }

  if (0 == tid) {
    atomicMinf(d_min, min_data[0]);
    atomicMaxf(d_max, max_data[0]);
  }
}

// Suboptimal implementation, should be reduce and scan to reduce number of
// atomic calls
__global__ void luminance_histogram(const float* const d_array,
                                    unsigned int* const d_bins,
                                    const int numBins, float min_logLum,
                                    float range_logLum, uint size) {
  uint tid = threadIdx.x;
  int id = (blockDim.x * blockIdx.x) + tid;
  if (id >= size) {
    return;
  }
  float data = d_array[id];
  size_t bin =
      static_cast<size_t>(((data - min_logLum) / range_logLum) * numBins);
  bin = bin < numBins ? bin : numBins - 1;
  atomicAdd(&(d_bins[bin]), 1);
};

// Hillis Steele (exlusive)
__global__ void hillis_steele_scan(unsigned int* const d_cdf,
                                   const unsigned int* const d_in,
                                   const unsigned int step,
                                   const size_t numBins) {
  uint tid = threadIdx.x;
  int id = (blockDim.x * blockIdx.x) + tid;
  const int offset{1 << step};
  if (id >= numBins) {
    return;
  }
  if (id >= offset) {
    d_cdf[id] +=d_in[id - offset];
  }
};

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf, float& min_logLum,
                                  float& max_logLum, const size_t numRows,
                                  const size_t numCols, const size_t numBins) {
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  uint blockSize(512);
  uint gridSize((numCols * numRows + blockSize - 1) / blockSize);

  // 1
  float *d_min, *d_max;
  checkCudaErrors(cudaMalloc((void**)&d_min, sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_max, sizeof(float)));
  shmem_getExtrema<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(
      d_logLuminance, d_min, d_max, numCols * numRows);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(
      cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(
      cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_max));

  // 2
  printf("Cuda log: \n min: %f\n max: %f \n", min_logLum, max_logLum);
  float range_logLum = max_logLum - min_logLum;

  // 3
  const int BIN_BYTES = numBins * sizeof(unsigned int);
  int h_bins[numBins];
  for (size_t i = 0; i < numBins; i++) {
    h_bins[i] = 0;
  }
  unsigned int* d_bins;
  cudaMalloc((void**)&d_bins, BIN_BYTES);
  cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice);

  luminance_histogram<<<gridSize, blockSize>>>(d_logLuminance, d_bins, numBins,
                                               min_logLum, range_logLum,
                                               numCols * numRows);
  cudaDeviceSynchronize();
  cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);
  // for (auto i = 1014; i < 1024; ++i) {
  //   printf("Cuda log: bin %i: %i \n", i, h_bins[i]);
  // }
  // auto sum = 0U;
  // for (auto i = 0; i < 1024; ++i) {
  //   sum+=h_bins[i];
  // }
  // printf("Cuda number of items: %i\n", sum);

  // 4
  auto num_steps = static_cast<int>(std::log2(numBins));
  // auto num_steps = 1;
  auto scanGridSize = (numBins + blockSize - 1) / blockSize;
  printf("Number of scan steps: %i \n", num_steps);
  checkCudaErrors(
      cudaMemcpy(&d_cdf[1], d_bins, BIN_BYTES-sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  for (auto k = 0; k < num_steps; ++k) {
    checkCudaErrors(
        cudaMemcpy(d_bins, d_cdf, BIN_BYTES, cudaMemcpyDeviceToDevice));
    hillis_steele_scan<<<scanGridSize, blockSize>>>(d_cdf, d_bins, k, numBins);
  }
  cudaMemcpy(h_bins, d_cdf, BIN_BYTES, cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaFree(d_bins));
  cudaDeviceSynchronize();
}
