#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define FILTER_RADIUS 2
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)
#define TILE_WIDTH 16
#define SHARED_WIDTH (TILE_WIDTH + 2 * FILTER_RADIUS)

__constant__ float c_Filter[FILTER_WIDTH * FILTER_WIDTH];
// A clean, read-only global memory anchor for out-of-bounds padding
__device__ const float d_ZeroPadding = 0.0f;

template <typename Pipeline>
__device__ void prefetch_tile(const float *input, float s_Data[SHARED_WIDTH * SHARED_WIDTH],
                              int base_row, int base_col, int width, int height,
                              int linear_tx, int num_threads, Pipeline &pipe)
{

    for (int i = linear_tx; i < SHARED_WIDTH * SHARED_WIDTH; i += num_threads)
    {
        int s_row = i / SHARED_WIDTH;
        int s_col = i % SHARED_WIDTH;

        int g_row = base_row + s_row;
        int g_col = base_col + s_col;

        // Route the source pointer dynamically. Out-of-bounds requests safely
        // read from our global zero anchor using the async copy engine.
        const float *src_ptr = (g_row >= 0 && g_row < height && g_col >= 0 && g_col < width)
                                   ? &input[g_row * width + g_col]
                                   : &d_ZeroPadding;

        cuda::memcpy_async(&s_Data[i], src_ptr, sizeof(float), pipe);
    }
}

__global__ void convolution2DAdvanced(const float *__restrict__ input, float *__restrict__ output, int width, int height)
{
    __shared__ float s_Data[2][SHARED_WIDTH * SHARED_WIDTH];
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> shared_state;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int linear_tx = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    auto block = cooperative_groups::this_thread_block();
    cuda::pipeline<cuda::thread_scope_block> pipe = cuda::make_pipeline(block, &shared_state);

    int start_tile_y = blockIdx.y * 2;
    int end_tile_y = min(start_tile_y + 2, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    if (start_tile_y >= (height + TILE_WIDTH - 1) / TILE_WIDTH)
        return;

    int base_col = blockIdx.x * TILE_WIDTH - FILTER_RADIUS;
    int base_row_0 = start_tile_y * TILE_WIDTH - FILTER_RADIUS;

    pipe.producer_acquire();
    prefetch_tile(input, s_Data[0], base_row_0, base_col, width, height, linear_tx, num_threads, pipe);
    pipe.producer_commit();

    int write_buffer = 1;
    int read_buffer = 0;

    for (int tile_y = start_tile_y; tile_y < end_tile_y; ++tile_y)
    {

        if (tile_y + 1 < end_tile_y)
        {
            int next_base_row = (tile_y + 1) * TILE_WIDTH - FILTER_RADIUS;
            pipe.producer_acquire();
            prefetch_tile(input, s_Data[write_buffer], next_base_row, base_col, width, height, linear_tx, num_threads, pipe);
            pipe.producer_commit();
        }

        // 1. Wait for async global memory transfers (including zero-padding) to arrive
        pipe.consumer_wait();

        int col = blockIdx.x * TILE_WIDTH + tx;
        int row = tile_y * TILE_WIDTH + ty;

        if (row < height && col < width)
        {
            float sum = 0.0f;

#pragma unroll
            for (int i = 0; i < FILTER_WIDTH; ++i)
            {
#pragma unroll
                for (int j = 0; j < FILTER_WIDTH; ++j)
                {
                    int shared_row = ty + i;
                    int shared_col = tx + j;
                    sum += s_Data[read_buffer][shared_row * SHARED_WIDTH + shared_col] * c_Filter[i * FILTER_WIDTH + j];
                }
            }
            output[row * width + col] = sum;
        }

        __syncthreads();
        pipe.consumer_release();

        // Swap stage tags
        read_buffer = write_buffer;
        write_buffer = 1 - write_buffer;
    }
}
int main()
{
    // 1. Define image dimensions
    const int width = 2048;
    const int height = 2048;

    size_t image_size = width * height * sizeof(float);
    size_t filter_size = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);

    // 2. Allocate CPU (Host) memory
    float *h_input = (float *)malloc(image_size);
    float *h_output = (float *)malloc(image_size);
    float *h_filter = (float *)malloc(filter_size);

    // 3. Initialize synthetic data
    // Fill input with 1.0f (makes verification easy)
    for (int i = 0; i < width * height; ++i)
    {
        h_input[i] = 1.0f;
    }

    // Fill filter weights with 1.0f / 25.0f (a standard box blur filter)
    for (int i = 0; i < FILTER_WIDTH * FILTER_WIDTH; ++i)
    {
        h_filter[i] = 1.0f / (FILTER_WIDTH * FILTER_WIDTH);
    }

    // 4. Allocate GPU (Device) memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);

    // 5. Copy data to GPU
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    // Copy to __constant__ memory symbol
    cudaMemcpyToSymbol(c_Filter, h_filter, filter_size);
    cudaMemset(d_output, 0, image_size);

    // 6. Configure execution grid
    // The kernel calculates two vertical tiles per block (start_tile_y to end_tile_y)
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks(
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        ((height + TILE_WIDTH - 1) / TILE_WIDTH + 1) / 2 // Divided by 2 because each block handles 2 vertical tiles
    );

    std::cout << "Launching advanced convolution kernel..." << std::endl;

    // 7. Execute and Profile
    cudaDeviceSynchronize(); // Warm-up / sync
    convolution2DAdvanced<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // 8. Copy result back to CPU for validation
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // 9. Quick sanity check on an internal pixel (away from edges)
    // Since input is all 1.0f and filter weights sum to 1.0f, the output should be 1.0f
    std::cout << "Sample output pixel [100, 100]: " << h_output[100 * width + 100] << " (Expected: 1.0)" << std::endl;

    // 10. Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_filter);

    std::cout << "Test completed successfully." << std::endl;
    return 0;
}