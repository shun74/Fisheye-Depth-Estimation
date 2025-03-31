#include <cuda_runtime.h>

extern "C"
{
    void calc_points(const float *disp, const float *map_x, const float *map_y, int w, int h, float fx, float fy,
                     float cx, float cy, float base_line, float *pcd_ptr, bool *valid);
}
