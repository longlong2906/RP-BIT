foreach(name native_paper_v4_protocol native_paper_v4_worker native_paper_v4_storage
             native_paper_v4_analysis native_paper_v4_campaign native_paper_v4_resume kaggle_native_paper_v4)
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_${name}.py)
  set_tests_properties(${name} PROPERTIES LABELS native_paper_v4)
endforeach()
