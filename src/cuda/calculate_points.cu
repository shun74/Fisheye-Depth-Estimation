#include "calculate_points.h"

__global__ void calc_points_kernel(const float *disp, const float *map_x, const float *map_y, int w, int h, float fx,
                                   float fy, float cx, float cy, float base_line, float *pcd_ptr, bool *valid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int px = idx % w;
    int py = idx / w;

    if (px >= w || py >= h)
        return;

    float d = disp[py * w + px] / 16.0f;
    int disp_x = __float2int_rd(px - d);

    float lx = map_x[py * w + px];
    float rx = (disp_x >= 0 && disp_x < w) ? map_x[py * w + disp_x] : 0.0f;
    float adjusted_d = lx - rx;

    float cy_val = map_y[py * w + px];

    float z = fx * base_line / adjusted_d;
    float x = z * (lx - cx) / fx;
    float y = z * (cy_val - cy) / fy;

    bool point_valid = (px >= 0) && (py >= 0) && (d >= 1.0f) && (disp_x >= 0) && (disp_x < w) &&
                       (fabsf(adjusted_d) > 1e-6) && isfinite(x) && isfinite(y) && isfinite(z);

    pcd_ptr[3 * (py * w + px) + 0] = point_valid ? x : 0.0f;
    pcd_ptr[3 * (py * w + px) + 1] = point_valid ? y : 0.0f;
    pcd_ptr[3 * (py * w + px) + 2] = point_valid ? z : 0.0f;
    valid[py * w + px] = point_valid;
}

extern "C" void calc_points(const float *disp, const float *map_x, const float *map_y, int w, int h, float fx, float fy,
                            float cx, float cy, float base_line, float *pcd_ptr, bool *valid)
{
    int block_size = 32;
    int grid_size = (w * h + block_size - 1) / block_size;

    calc_points_kernel<<<grid_size, block_size>>>(disp, map_x, map_y, w, h, fx, fy, cx, cy, base_line, pcd_ptr, valid);
    cudaDeviceSynchronize();
}
