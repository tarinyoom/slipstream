cmake_minimum_required(VERSION 3.24)
project(slipstream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SLIPSTREAM_BUILD_PYTHON "Build the nanobind Python extension" OFF)

find_package(GTest CONFIG REQUIRED)

add_library(slipstream_core SHARED
    lib/src/state.cpp
    lib/src/solver.cpp
    lib/src/cpu/advect.cpp
    lib/src/cpu/project.cpp
)
target_include_directories(slipstream_core PUBLIC lib/include)
target_compile_options(slipstream_core PRIVATE -Wall -Wextra)

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
)
target_include_directories(slipstream_tests PRIVATE lib/src)
target_link_libraries(slipstream_tests PRIVATE slipstream_core GTest::gtest_main)

enable_testing()
add_test(NAME slipstream_tests COMMAND slipstream_tests)
