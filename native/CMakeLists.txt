cmake_minimum_required(VERSION 3.24)
project(hex_transcribe_shim CXX)

# Builds the small C-ABI shim Odin loads at runtime (core:dynlib).
#
# The heavy half — ggml + transcribe.cpp + Vulkan shaders — is built once, on
# its own, because it is slow and its CMake assumes it owns the source root:
#
#   cmake -S native/third_party/transcribe.cpp -B native/build-transcribe \
#         -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF
#   cmake --build native/build-transcribe --config Release --parallel
#
# Then this project links the resulting static libs. Set HEX_FETCH_TRANSCRIBE=ON
# to fetch the pinned revision instead of using native/third_party.

set(HEX_ROOT        "${CMAKE_CURRENT_SOURCE_DIR}")
set(HEX_TRANSCRIBE  "${HEX_ROOT}/third_party/transcribe.cpp")
set(HEX_TRANSCRIBE_BUILD "${HEX_ROOT}/build-transcribe")
set(TRANSCRIBE_REV  "7045d1c0fe7cee37747b2792977421d435e75fca" CACHE STRING "anomalyco/transcribe.cpp revision")
option(HEX_FETCH_TRANSCRIBE "Fetch transcribe.cpp instead of using third_party/" OFF)

add_library(transcribe_shim SHARED transcribe_shim.cpp)
set_target_properties(transcribe_shim PROPERTIES
  OUTPUT_NAME transcribe_shim
  RUNTIME_OUTPUT_DIRECTORY "${HEX_ROOT}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${HEX_ROOT}/bin"
  # ggml is built with the dynamic CRT (/MD); match it or the link collides.
  MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
)

if (HEX_FETCH_TRANSCRIBE)
  include(FetchContent)
  FetchContent_Declare(transcribe_cpp
    GIT_REPOSITORY https://github.com/anomalyco/transcribe.cpp
    GIT_TAG ${TRANSCRIBE_REV})
  set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(transcribe_cpp)
  target_include_directories(transcribe_shim PRIVATE ${transcribe_cpp_SOURCE_DIR}/include)
  target_link_libraries(transcribe_shim PRIVATE transcribe ggml ggml-cpu ggml-base advapi32)
  target_compile_definitions(transcribe_shim PRIVATE HEX_HAVE_TRANSCRIBE=1 TRANSCRIBE_STATIC=1)
  return()
endif()

if (EXISTS "${HEX_TRANSCRIBE}/include/transcribe.h")
  target_include_directories(transcribe_shim PRIVATE
    "${HEX_TRANSCRIBE}/include"
    "${HEX_TRANSCRIBE}/ggml/include")

  set(HEX_LIB_DIRS
    "${HEX_TRANSCRIBE_BUILD}/src/Release"
    "${HEX_TRANSCRIBE_BUILD}/ggml/src/Release")
  target_link_directories(transcribe_shim PRIVATE ${HEX_LIB_DIRS})

  if (EXISTS "${HEX_TRANSCRIBE_BUILD}/src/Release/transcribe.lib")
    target_link_libraries(transcribe_shim PRIVATE transcribe ggml ggml-cpu ggml-base advapi32)
    target_compile_definitions(transcribe_shim PRIVATE HEX_HAVE_TRANSCRIBE=1 TRANSCRIBE_STATIC=1)
    message(STATUS "transcribe_shim: linking prebuilt transcribe.cpp (CPU)")
  else()
    message(WARNING "transcribe_shim: transcribe.lib not built yet — shim will be a stub")
  endif()
else()
  message(WARNING "transcribe_shim: no transcribe.cpp sources — shim will be a stub")
endif()
