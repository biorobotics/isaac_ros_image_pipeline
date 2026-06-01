extern "C" __global__
void detect_laser_stripe_kernel_ordered(
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
