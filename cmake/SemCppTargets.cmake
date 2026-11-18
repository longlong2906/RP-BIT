# C++ port of the project's SEM-RRT* reconstruction, in its own library so the existing
# terrain_compare_c bridge (and every campaign bound to its hash) stays unchanged.
add_library(sem_rrtstar_c SHARED src/SemRrtStar.cpp src/SemRrtStarC.cpp)
target_compile_definitions(sem_rrtstar_c PRIVATE SEM_C_EXPORTS)
target_link_libraries(sem_rrtstar_c PRIVATE terrain25d)
if(NOT MSVC)
  # Bit-for-bit agreement with CPython needs the same operation order: no FMA contraction.
  target_compile_options(sem_rrtstar_c PRIVATE -ffp-contract=off -fno-fast-math -Wall -Wextra -Wpedantic)
endif()
add_test(NAME sem_native_cpp COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_sem_native_cpp.py
  --library $<TARGET_FILE:sem_rrtstar_c> --bridge $<TARGET_FILE:terrain_compare_c>)
set_tests_properties(sem_native_cpp PROPERTIES LABELS sem_cpp RUN_SERIAL TRUE TIMEOUT 3600)
