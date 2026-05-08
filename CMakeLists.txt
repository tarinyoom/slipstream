cmake_minimum_required(VERSION 3.24)
project(slipstream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SLIPSTREAM_BUILD_PYTHON "Build the nanobind Python extension" OFF)
option(SLIPSTREAM_BUILD_CUDA   "Build the CUDA GPU backend"          OFF)

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

add_library(slipstream_core SHARED
    lib/src/arena.cpp
    lib/src/cpu/compute_advection.cpp
    lib/src/cpu/compute_projection.cpp
    lib/src/cpu/compute_injection.cpp
    lib/src/cpu/compute_buoyancy.cpp
    lib/src/cpu/compute_cooling.cpp
    lib/src/cpu/step.cpp
)
target_include_directories(slipstream_core PUBLIC lib/include)
target_include_directories(slipstream_core PRIVATE lib/src)
target_compile_options(slipstream_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

if(SLIPSTREAM_BUILD_CUDA)
    target_sources(slipstream_core PRIVATE
        lib/src/gpu/memory.cpp
        lib/src/gpu/transfer.cpp
        lib/src/gpu/compute_advection.cu
        lib/src/gpu/compute_injection.cu
        lib/src/gpu/compute_buoyancy.cu
        lib/src/gpu/compute_cooling.cu
        lib/src/gpu/compute_projection.cu
        lib/src/gpu/step.cu
    )
    target_compile_definitions(slipstream_core PUBLIC SLIPSTREAM_HAS_CUDA=1)
    target_link_libraries(slipstream_core PRIVATE CUDA::cudart)
    set_target_properties(slipstream_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(SLIPSTREAM_BUILD_PYTHON)
    find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core python/slipstream_py.cpp)
    target_link_libraries(_core PRIVATE slipstream_core)
    set_target_properties(_core PROPERTIES
        BUILD_RPATH  "$<TARGET_FILE_DIR:slipstream_core>"
        INSTALL_RPATH "$ORIGIN"
    )
    install(TARGETS _core           LIBRARY DESTINATION slipstream)
    install(TARGETS slipstream_core LIBRARY DESTINATION slipstream)
endif()

add_executable(slipstream_tests
    tests/cpp/physics/test_incompressibility.cpp
    tests/cpp/physics/test_scalar_transport.cpp
    tests/cpp/physics/test_plume_behavior.cpp
)
target_include_directories(slipstream_tests PRIVATE lib/src)
target_link_libraries(slipstream_tests PRIVATE slipstream_core GTest::gtest_main)

if(SLIPSTREAM_BUILD_CUDA)
    add_executable(slipstream_tests_gpu tests/cpp/cuda_parity/test_gpu_parity.cu)
    target_include_directories(slipstream_tests_gpu PRIVATE lib/src)
    target_link_libraries(slipstream_tests_gpu PRIVATE slipstream_core GTest::gtest_main)
endif()

add_executable(slipstream tools/cli.cpp)
target_include_directories(slipstream PRIVATE lib/include lib/src)
target_link_libraries(slipstream PRIVATE slipstream_core)

enable_testing()
gtest_discover_tests(slipstream_tests)
if(SLIPSTREAM_BUILD_CUDA)
    gtest_discover_tests(slipstream_tests_gpu)
endif()
