#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
# if defined(SEM_C_EXPORTS)
#  define SEM_API __declspec(dllexport)
# else
#  define SEM_API __declspec(dllimport)
# endif
#else
# define SEM_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* C++ port of the project's SEM-RRT* reconstruction. Handles are process-local and not
   thread-safe; errors never cross this ABI (-1 / NULL plus sem_error()). */
SEM_API const char *sem_error(void);
SEM_API const char *sem_version(void);
/* Loads the terrain (collision index on, as tc_create(path, 0)) and copies the native raster:
   row-major [y][x], valid as 0/1 bytes, risk float64 with NaN where invalid. */
SEM_API void *sem_create(const char *map_path, const double bounds4[4], int nx, int ny, double resolution,
                         const unsigned char *valid, const double *risk);
SEM_API void sem_destroy(void *planner);
/* Return nonzero to abort the solve. heights holds the terrain height of each path vertex. */
typedef int (*sem_callback)(double elapsed_seconds, const double *path_xy, const double *heights,
                            size_t point_count, double cost, const double *totals3, void *user);
/* Status: 0 EXACT_SOLUTION, 1 TIMEOUT, 2 INVALID_HARD_ENDPOINTS, 3 INVALID_NATIVE_ENDPOINTS,
   4 aborted by the callback, -1 error. already_elapsed_s is charged to the budget. */
SEM_API int sem_solve(void *planner, double sx, double sy, double gx, double gy, uint64_t seed,
                      double budget_seconds, uint64_t work_iterations, double already_elapsed_s,
                      sem_callback callback, void *user);
/* iterations, sample_proposals, sampler_failures, tree_insertions, edge_evaluations,
   hard_rejections, native_rejections, parent_candidates, rewires, descendant_cost_updates,
   start_connection_checks, incumbents, late_candidates, nodes */
SEM_API int sem_counters(void *planner, uint64_t out14[14]);
SEM_API int sem_set_exhaustive_check(void *planner, int enabled);
/* Optional convergence stop for the following solves; window 0 turns it off (default). Returns 1. */
SEM_API int sem_set_convergence_stop(void *planner, double relative_eps, uint64_t window_iterations);
/* Last solve: returns 1 if it stopped on convergence, 0 otherwise, -1 error. out2 = reference cost
   (NaN before the first solution) and the iteration count at the last reference reset. */
SEM_API int sem_convergence(void *planner, double out2[2]);
/* Equivalence-test helpers. */
SEM_API int sem_model_edge(void *planner, double ax, double ay, double bx, double by, double out2[2]);
SEM_API int sem_model_valid_state(void *planner, double x, double y);
SEM_API int sem_debug_random(uint64_t seed, size_t count, double *out);
SEM_API int sem_debug_uniform(uint64_t seed, double low, double high, size_t count, double *out);
SEM_API double sem_debug_hypot(double x, double y);
SEM_API double sem_debug_dist(double ax, double ay, double bx, double by);
SEM_API double sem_debug_fsum(const double *values, size_t count);
SEM_API double sem_debug_turn(double ax, double ay, double bx, double by, double cx, double cy);
/* out4 = cos(x), sin(x), atan2(y, x), acos(z) */
SEM_API int sem_debug_libm(double x, double y, double z, double *out4);
#ifdef __cplusplus
}
#endif
