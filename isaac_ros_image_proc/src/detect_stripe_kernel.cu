#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// // Row-wise laser stripe detection (new version)
// extern "C"
// __global__ void detect_laser_stripe_kernel(
//     const unsigned char* __restrict__ im,
//     int rows, int cols, size_t step,
//     float roi_x, float roi_y,
//     float threshold_factor,
//     float* out_x, float* out_y,
//     int*  out_count)
// {
//     int rr = blockIdx.x * blockDim.x + threadIdx.x;
//     if (rr >= rows) return;

//     const unsigned char* row_ptr = im + rr * step;
//     unsigned char max_val = 0;
//     int max_col = 0;
//     for (int cc = 0; cc < cols; ++cc) {
//         unsigned char v = row_ptr[cc];
//         if (v > max_val) {
//             max_val = v;
//             max_col = cc;
//         }
//     }

//     float thr = threshold_factor * max_val;
//     int i = max_col - 1;
//     while (i >= 0 && row_ptr[i] > thr) --i;
//     int k = max_col + 1;
//     while (k < cols && row_ptr[k] > thr) ++k;

//     double weighted_sum = 0.0, val_sum = 0.0;
//     for (int cc = i + 1; cc < k; ++cc) {
//         double v = row_ptr[cc];
//         weighted_sum += v * (cc - i);
//         val_sum      += v;
//     }

//     if (val_sum > 0.0) {
//         float x = float(weighted_sum / val_sum + i) + roi_x;
//         float y = float(rr) + roi_y;
//         int idx = atomicAdd(out_count, 1);
//         out_x[idx] = x;
//         out_y[idx] = y;
//     }
// }


// // Kernel: one thread per column
// extern "C"
// __global__ void detect_laser_stripe_kernel(
//     const unsigned char* __restrict__ im,  // pointer to image data
//     int rows, int cols, size_t step,       // dims + row-pitch
//     float roi_x, float roi_y,              // ROI offset
//     float threshold_factor,                // e.g. 0.8f
//     float* out_x, float* out_y,            // output arrays (size ≥ cols)
//     int*  out_count)                       // atomic counter
// {
//     int cc = blockIdx.x * blockDim.x + threadIdx.x;
//     if (cc >= cols) return;

//     // pointer to start of this column
//     const unsigned char* col_ptr = im + cc;

//     // 1) find max
//     unsigned char max_val = 0;
//     int max_row = 0;
//     for (int rr = 0; rr < rows; ++rr) {
//         unsigned char v = col_ptr[rr * step];
//         if (v > max_val) {
//             max_val = v;
//             max_row = rr;
//         }
//     }

//     // 2) walk up/down where value > threshold_factor * max_val
//     float thr = threshold_factor * max_val;
//     int j = max_row - 1;
//     while (j >= 0 && col_ptr[j * step] > thr) --j;
//     int k = max_row + 1;
//     while (k < rows && col_ptr[k * step] > thr) ++k;

//     // 3) weighted centroid in [j+1, k)
//     double weighted_sum = 0.0, val_sum = 0.0;
//     for (int rr = j + 1; rr < k; ++rr) {
//         double v = col_ptr[rr * step];
//         weighted_sum += v * (rr - j);
//         val_sum      += v;
//     }

//     // 4) if valid, compute world coords & append
//     if (val_sum > 0.0) {
//         float y = float(weighted_sum / val_sum + j) + roi_y;
//         float x = float(cc) + roi_x;
//         int idx = atomicAdd(out_count, 1);
//         out_x[idx] = x;
//         out_y[idx] = y;
//     }
// }

extern "C" __global__
void detect_laser_stripe_kernel(
    const unsigned char* __restrict__ im,  // CV_8UC1 device pointer
    int rows, int cols, size_t step_bytes, // step in BYTES (Mat.step)
    float roi_x, float roi_y,
    float threshold_factor,                // e.g., 0.8f
    float* __restrict__ out_x,             // size = cols
    float* __restrict__ out_y,             // size = cols
    unsigned char* __restrict__ valid)     // size = cols, 0/1
{
    int cc = blockIdx.x * blockDim.x + threadIdx.x;
    if (cc >= cols) return;

    const unsigned char* col0 = im + cc;   // first row, column cc

    // 1) max search in this column
    unsigned char max_val = 0;
    int max_row = 0;
    for (int rr = 0; rr < rows; ++rr) {
        unsigned char v = col0[rr * step_bytes];
        if (v > max_val) { max_val = v; max_row = rr; }
    }

    // 2) threshold run
    float thr = threshold_factor * float(max_val);
    int j = max_row - 1; while (j >= 0    && col0[j * step_bytes] > thr) --j;
    int k = max_row + 1; while (k < rows && col0[k * step_bytes] > thr) ++k;

    // 3) centroid on (j+1 .. k-1)
    double wsum = 0.0, vsum = 0.0;
    for (int rr = j + 1; rr < k; ++rr) {
        double v = double(col0[rr * step_bytes]);
        wsum += v * (rr - j);
        vsum += v;
    }

    if (vsum > 0.0) {
        out_x[cc] = float(cc) + roi_x;
        out_y[cc] = float(wsum / vsum + j) + roi_y;
        valid[cc] = 1;
    } else {
        valid[cc] = 0; // no point in this column
    }
}
