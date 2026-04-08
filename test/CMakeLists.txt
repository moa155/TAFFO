add_subdirectory(0_simple)
add_subdirectory(1_c++)
#add_subdirectory(2_openmp)
add_subdirectory(fpbench)
add_subdirectory(polybench)

# Donut range self-test (sub-project #1). Only added when the TaffoCommon
# library target exists in this configuration; otherwise it is safely
# skipped so we do not block the main build.
if(TARGET TaffoCommon)
  add_subdirectory(donut_ranges)
endif()
