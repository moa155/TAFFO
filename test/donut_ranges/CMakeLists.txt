# Donut range self-test. This is a standalone binary that links against
# the TaffoCommon library and exercises the donut-range extensions of
# taffo::Range directly, without depending on the stale gtest
# infrastructure under unittests/.
#
# To run:
#   cmake --build build --target donut_range_selftest
#   ./build/test/donut_ranges/donut_range_selftest

add_executable(donut_range_selftest donut_range_selftest.cpp)
target_link_libraries(donut_range_selftest PRIVATE TaffoCommon)

# Micro-benchmark exercising the full VRA arithmetic. We do NOT link
# against obj.TaffoVRA because that object library pulls in
# CodeInterpreter.cpp which references llvm::Value::dump(), a symbol
# that is only exported from LLVM when it was built with
# LLVM_ENABLE_DUMP=ON (i.e. debug builds). Homebrew's release LLVM
# ships without it, causing the link to fail.
#
# Instead, we compile the two source files we actually need directly
# into the microbench executable: RangeOperations.cpp (the arithmetic
# operators) and RangeOperationsCallWhitelist.cpp (the math/activation
# function handlers). Both depend only on TaffoCommon + base LLVM
# headers that ARE available in any LLVM distribution.
add_executable(donut_microbench
  donut_microbench.cpp
  ${CMAKE_SOURCE_DIR}/lib/TaffoVRA/TaffoVRA/RangeOperations.cpp
  ${CMAKE_SOURCE_DIR}/lib/TaffoVRA/TaffoVRA/RangeOperationsCallWhitelist.cpp)
target_include_directories(donut_microbench PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/TaffoVRA
  ${CMAKE_SOURCE_DIR}/lib/TypeDeductionAnalysis/src)
target_link_libraries(donut_microbench PRIVATE TaffoCommon)

# Arithmetic unit tests: link the same two VRA source files as the
# microbench target so we can call handleAdd/Sub/Mul/Div directly.
add_executable(donut_arith_test
  donut_arith_test.cpp
  ${CMAKE_SOURCE_DIR}/lib/TaffoVRA/TaffoVRA/RangeOperations.cpp
  ${CMAKE_SOURCE_DIR}/lib/TaffoVRA/TaffoVRA/RangeOperationsCallWhitelist.cpp)
target_include_directories(donut_arith_test PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/TaffoVRA
  ${CMAKE_SOURCE_DIR}/lib/TypeDeductionAnalysis/src)
target_link_libraries(donut_arith_test PRIVATE TaffoCommon)

# TaffoCommon already exposes its include path via target_include_directories
# PUBLIC, so we do not need to add anything else. If the main build does not
# define TaffoCommon for some reason (e.g. a stripped-down configuration),
# exclude this target.
