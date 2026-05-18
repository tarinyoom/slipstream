cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT is not set. This project requires vcpkg.")
endif()

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain (auto-set from VCPKG_ROOT)")
endif()

if(NOT DEFINED SLIPSTREAM_BUILD_OPENVDB)
    set(SLIPSTREAM_BUILD_OPENVDB ON CACHE BOOL "Build CLI with OpenVDB output")
endif()

if(NOT DEFINED SLIPSTREAM_BUILD_CUDA)
    set(SLIPSTREAM_BUILD_CUDA ON CACHE BOOL "Build the CUDA GPU backend")
endif()

if(SLIPSTREAM_BUILD_OPENVDB AND NOT DEFINED VCPKG_MANIFEST_FEATURES)
    set(VCPKG_MANIFEST_FEATURES "openvdb"
        CACHE STRING "vcpkg manifest features (auto-set by SLIPSTREAM_BUILD_OPENVDB)")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(slipstream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(SLIPSTREAM_BUILD_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_library(slipstream_core STATIC
    lib/src/state.cpp
    lib/src/memory.cpp
    lib/src/k_cpu/advection.cpp
    lib/src/k_cpu/projection.cpp
    lib/src/k_cpu/injection.cpp
    lib/src/k_cpu/buoyancy.cpp
    lib/src/k_cpu/cooling.cpp
    lib/src/step.cpp
)
target_include_directories(slipstream_core PUBLIC lib/include)
target_include_directories(slipstream_core PRIVATE lib/src)
target_compile_options(slipstream_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    >)

if(SLIPSTREAM_BUILD_CUDA)
    target_sources(slipstream_core PRIVATE
        lib/src/k_cuda/advection.cu
        lib/src/k_cuda/injection.cu
        lib/src/k_cuda/buoyancy.cu
        lib/src/k_cuda/cooling.cu
        lib/src/k_cuda/projection.cu
    )
    target_compile_definitions(slipstream_core PUBLIC SLIPSTREAM_HAS_CUDA=1)
    target_link_libraries(slipstream_core PRIVATE CUDA::cudart)
    set_target_properties(slipstream_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(slipstream_tests
    tests/cpp/physics/test_incompressibility.cpp
    tests/cpp/physics/test_scalar_transport.cpp
    tests/cpp/physics/test_plume_behavior.cpp
)
target_include_directories(slipstream_tests PRIVATE lib/src)
target_link_libraries(slipstream_tests PRIVATE slipstream_core GTest::gtest_main)

if(SLIPSTREAM_BUILD_CUDA)
    add_executable(slipstream_tests_cuda tests/cpp/cuda_parity/test_cuda_parity.cu)
    target_include_directories(slipstream_tests_cuda PRIVATE lib/src)
    target_link_libraries(slipstream_tests_cuda PRIVATE slipstream_core GTest::gtest_main)
endif()

add_executable(slipstream tools/cli.cpp)
target_include_directories(slipstream PRIVATE lib/include tools)
target_link_libraries(slipstream PRIVATE slipstream_core)
if(SLIPSTREAM_BUILD_CUDA)
    target_link_libraries(slipstream PRIVATE CUDA::cudart)
endif()
if(SLIPSTREAM_BUILD_OPENVDB)
    find_package(OpenVDB CONFIG REQUIRED)
    find_package(blosc CONFIG REQUIRED)
    find_package(ZLIB REQUIRED)
    target_sources(slipstream PRIVATE tools/vdb.cpp)
    target_link_libraries(slipstream PRIVATE
        OpenVDB::openvdb blosc_static ZLIB::ZLIB)
    target_compile_definitions(slipstream PRIVATE SLIPSTREAM_HAS_OPENVDB=1)
endif()

enable_testing()
gtest_discover_tests(slipstream_tests)
if(SLIPSTREAM_BUILD_CUDA)
    gtest_discover_tests(slipstream_tests_cuda)
endif()
