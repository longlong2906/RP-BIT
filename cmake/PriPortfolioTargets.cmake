# Opt-in real BIT* portfolio; legacy objective and backend paths are unchanged.
if(TERRAIN_WITH_OMPL)
  target_sources(terrain_ompl PRIVATE src/PriPortfolio.cpp)
  add_executable(pri_portfolio_tests tests/integration/pri_portfolio_tests.cpp)
  target_link_libraries(pri_portfolio_tests PRIVATE terrain_ompl)
  add_test(NAME pri_portfolio_core COMMAND pri_portfolio_tests)
  set_tests_properties(pri_portfolio_core PROPERTIES LABELS pri_portfolio TIMEOUT 180)
  if(WIN32)
    set_tests_properties(pri_portfolio_core PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CMAKE_SOURCE_DIR}/.deps/install/lib;PATH=path_list_prepend:C:/mingw64/bin")
  endif()
  add_test(NAME pri_portfolio_worker COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_pri_portfolio_worker.py --worker $<TARGET_FILE:terrain_benchmark>)
  set_tests_properties(pri_portfolio_worker PROPERTIES LABELS pri_portfolio RUN_SERIAL TRUE TIMEOUT 600)
endif()
foreach(component protocol runner validation analysis sem_clock)
  add_test(NAME pri_portfolio_${component} COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_pri_portfolio_${component}.py)
  set_tests_properties(pri_portfolio_${component} PROPERTIES LABELS pri_portfolio RUN_SERIAL TRUE)
endforeach()
