#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
# if defined(TERRAIN_C_EXPORTS)
#  define TC_API __declspec(dllexport)
# else
#  define TC_API __declspec(dllimport)
# endif
#else
# define TC_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Handles are process-local, not thread-safe. Errors never cross this C ABI. */
TC_API void *tc_create(const char *map_path, int enable_risk);
TC_API void tc_destroy(void *terrain);
TC_API const char *tc_error(void);
TC_API int tc_valid_motion(void *terrain, double ax, double ay, double bx, double by);
TC_API int tc_edge_cost(void *terrain, double ax, double ay, double bx, double by, double out5[5]);
TC_API int tc_height(void *terrain, double x, double y, double *z);
TC_API int tc_bounds(void *terrain, double out4[4]);
/* Disjoint wall seconds: load including core preprocessing, risk, collision index. */
TC_API int tc_timings(void *terrain, double out3[3]);
/* Model conversion/voxel creation occurs here, before the solve timer. */
TC_API void *tc_putn_create(void *terrain);
TC_API void *tc_putn_create_paper(void *terrain, const double *cloud_xyz, size_t point_count);
/* Snapshot rows: COMxyz,normalxyz,tau,flatness,slope_deg,sparsity,point_count. */
TC_API int tc_putn_snapshot_features(void *planner, double *out, size_t row_capacity);
TC_API int tc_putn_fit_plane(void *planner, double x, double y, double out12[12]);
TC_API int tc_putn_paper_stats(void *planner, double out2[2]);
TC_API void tc_putn_destroy(void *planner);
typedef void (*tc_putn_callback)(double elapsed_seconds, const double *path_xy,
                                const double *raw_com_xyz, size_t point_count,
                                double planner_objective, void *user);
TC_API int tc_putn_solve(void *planner, double sx, double sy, double gx, double gy,
                        uint64_t seed, double budget_seconds, int own_cost,
                        uint64_t fixed_work, tc_putn_callback callback, void *user);
/* iterations,plane_fits,model_rejections,common_rejections,edge_cost_calls,
   rewires,improvements,nodes,voxel_preprocessing_s,solve_s,tree_consistent,conversion_s */
TC_API int tc_putn_stats(void *planner, double out12[12]);
#ifdef __cplusplus
}
#endif
