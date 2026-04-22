# Locate clang-format >= 19 for the formatting checks.
#
# A missing or too-old clang-format is a developer-experience issue, not a
# build blocker: on Homebrew the default `clang-format` on PATH is 18 while a
# newer one lives under `/opt/homebrew/opt/llvm/bin`. We widen the search and
# downgrade the "not found" / "too old" cases from FATAL_ERROR to STATUS so
# that configuration succeeds on developer machines without a bleeding-edge
# clang-format. The formatting job in CI still enforces the real check.
find_program(CLANG_FORMAT_EXECUTABLE
  NAMES clang-format-19 clang-format-20 clang-format-21 clang-format-22 clang-format
  HINTS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin
)

if(NOT CLANG_FORMAT_EXECUTABLE)
  message(STATUS "clang-format not found; formatting checks will be skipped.")
  return()
endif()

execute_process(
  COMMAND ${CLANG_FORMAT_EXECUTABLE} --version
  OUTPUT_VARIABLE CLANG_FORMAT_VERSION_OUTPUT
  ERROR_VARIABLE CLANG_FORMAT_VERSION_ERROR
  RESULT_VARIABLE CLANG_FORMAT_VERSION_RESULT
)

if(NOT CLANG_FORMAT_VERSION_RESULT EQUAL 0)
  message(STATUS "Failed to get clang-format version; formatting checks will be skipped.")
  return()
endif()

string(REGEX MATCH "[0-9]+" CLANG_FORMAT_VERSION ${CLANG_FORMAT_VERSION_OUTPUT})

if(CLANG_FORMAT_VERSION VERSION_LESS "19")
  message(STATUS
    "clang-format ${CLANG_FORMAT_VERSION} found at ${CLANG_FORMAT_EXECUTABLE}; "
    "version 19+ is recommended. Formatting checks will be skipped.")
  set(CLANG_FORMAT_EXECUTABLE "" CACHE FILEPATH "" FORCE)
else()
  message(STATUS "clang-format version ${CLANG_FORMAT_VERSION} found at ${CLANG_FORMAT_EXECUTABLE}.")
endif()
