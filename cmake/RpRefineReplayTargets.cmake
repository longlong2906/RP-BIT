if(TERRAIN_WITH_OMPL)
  add_test(NAME rp_refine_replay_worker COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_rp_refine_replay_worker.py --worker $<TARGET_FILE:terrain_benchmark>)
  set_tests_properties(rp_refine_replay_worker PROPERTIES LABELS rp_bit RUN_SERIAL TRUE TIMEOUT 600)
endif()
