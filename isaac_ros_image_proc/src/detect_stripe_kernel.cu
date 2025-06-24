#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel: one thread per column
extern "C"
__global__ void detect_laser_stripe_kernel(
    const unsigned char* __restrict__ im,  // pointer to image data
    int rows, int cols, size_t step,       // dims + row-pitch
    float roi_x, float roi_y,              // ROI offset
    float threshold_factor,                // e.g. 0.8f
    float* out_x, float* out_y,            // output arrays (size ≥ cols)
    int*  out_count)                       // atomic counter
{
    int cc = blockIdx.x * blockDim.x + threadIdx.x;
    if (cc >= cols) return;

    // pointer to start of this column
    const unsigned char* col_ptr = im + cc;

    // 1) find max
    unsigned char max_val = 0;
    int max_row = 0;
    for (int rr = 0; rr < rows; ++rr) {
        unsigned char v = col_ptr[rr * step];
        if (v > max_val) {
            max_val = v;
            max_row = rr;
        }
    }

    // 2) walk up/down where value > threshold_factor * max_val
    float thr = threshold_factor * max_val;
    int j = max_row - 1;
    while (j >= 0 && col_ptr[j * step] > thr) --j;
    int k = max_row + 1;
    while (k < rows && col_ptr[k * step] > thr) ++k;

    // 3) weighted centroid in [j+1, k)
    double weighted_sum = 0.0, val_sum = 0.0;
    for (int rr = j + 1; rr < k; ++rr) {
        double v = col_ptr[rr * step];
        weighted_sum += v * (rr - j);
        val_sum      += v;
    }

    // 4) if valid, compute world coords & append
    if (val_sum > 0.0) {
        float y = float(weighted_sum / val_sum + j) + roi_y;
        float x = float(cc) + roi_x;
        int idx = atomicAdd(out_count, 1);
        out_x[idx] = x;
        out_y[idx] = y;
    }
}
