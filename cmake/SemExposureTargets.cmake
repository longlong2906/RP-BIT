# Opt-in SEM-risk exposure; keeps legacy length and traversability objectives intact.
if(TERRAIN_WITH_OMPL)
  target_sources(terrain_ompl PRIVATE src/SemExposure.cpp)
  add_executable(sem_exposure_ompl_tests tests/integration/sem_exposure_ompl_tests.cpp)
  target_link_libraries(sem_exposure_ompl_tests PRIVATE terrain_ompl)
  add_test(NAME sem_exposure_ompl COMMAND sem_exposure_ompl_tests)
  set_tests_properties(sem_exposure_ompl PROPERTIES LABELS sem_exposure TIMEOUT 120)
  if(WIN32)
    set_tests_properties(sem_exposure_ompl PROPERTIES ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CMAKE_SOURCE_DIR}/.deps/install/lib;PATH=path_list_prepend:C:/mingw64/bin")
  endif()
endif()
foreach(component protocol runner analysis validation)
  add_test(NAME sem_exposure_${component} COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_sem_exposure_${component}.py)
  set_tests_properties(sem_exposure_${component} PROPERTIES LABELS sem_exposure RUN_SERIAL TRUE)
endforeach()
if(TERRAIN_WITH_OMPL)
  add_test(NAME sem_exposure_worker COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/test_sem_exposure_worker.py --worker $<TARGET_FILE:terrain_benchmark>)
  set_tests_properties(sem_exposure_worker PROPERTIES LABELS sem_exposure RUN_SERIAL TRUE TIMEOUT 600)
endif()
