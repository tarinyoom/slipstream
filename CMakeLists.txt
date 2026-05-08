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

add_library(slipstream_core SHARED
    lib/src/arena.cpp
    lib/src/cpu/advect.cpp
    lib/src/cpu/project.cpp
    lib/src/cpu/inject.cpp
    lib/src/cpu/buoyancy.cpp
    lib/src/cpu/cooling.cpp
    lib/src/cpu/step.cpp
)
target_include_directories(slipstream_core PUBLIC lib/include)
target_include_directories(slipstream_core PRIVATE lib/src)
target_compile_options(slipstream_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

if(SLIPSTREAM_BUILD_CUDA)
    target_sources(slipstream_core PRIVATE
        lib/src/gpu/memory.cpp
        lib/src/gpu/transfer.cpp
        lib/src/gpu/advect.cu
        lib/src/gpu/inject.cu
        lib/src/gpu/buoyancy.cu
        lib/src/gpu/cooling.cu
        lib/src/gpu/project.cu
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
    tests/cpp/test_conservation.cpp
    tests/cpp/test_divergence.cpp
    tests/cpp/test_convergence.cpp
    tests/cpp/test_advection.cpp
    tests/cpp/test_ppm.cpp
    tests/cpp/test_buoyancy.cpp
)
target_include_directories(slipstream_tests PRIVATE lib/src)
target_link_libraries(slipstream_tests PRIVATE slipstream_core GTest::gtest_main)

if(SLIPSTREAM_BUILD_CUDA)
    add_executable(slipstream_tests_gpu tests/cpp/test_gpu_parity.cu)
    target_include_directories(slipstream_tests_gpu PRIVATE lib/src)
    target_link_libraries(slipstream_tests_gpu PRIVATE slipstream_core GTest::gtest_main)
    add_test(NAME slipstream_tests_gpu COMMAND slipstream_tests_gpu)
endif()

add_executable(slipstream tools/cli.cpp)
target_include_directories(slipstream PRIVATE lib/include lib/src)
target_link_libraries(slipstream PRIVATE slipstream_core)

enable_testing()
add_test(NAME slipstream_tests COMMAND slipstream_tests)
