# The three-backend gate deliberately has no GP dependency. Legacy gates remain.
foreach(name native_paper_v3_runtime native_paper_v3_protocol native_paper_v3_campaign
             shared_sem_evaluator native_paper_v3_analysis native_paper_v3_memory kaggle_native_paper_v3)
  if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_${name}.py")
    add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_${name}.py)
    set_tests_properties(${name} PROPERTIES LABELS native_paper_v3)
  endif()
endforeach()
set_tests_properties(terrain_unit traversability_unit flat_direct_certificate a3_preprocessing
  independent_geometry traversability_independent_oracle putn_static putn_paper terrain_capi
  native_session_policy native_paper_oracle sem_native_planner
  PROPERTIES LABELS native_paper_v3)
if(TERRAIN_WITH_OMPL)
  set_tests_properties(ompl_policies traversability_ompl_policies traversability_worker
    a3_sampler_cache a3_optimization_regression a3_perf_worker a3_worker_regression
    PROPERTIES LABELS native_paper_v3)
endif()
