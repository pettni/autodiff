cmake_minimum_required(VERSION 3.11)
project(autodiff_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)  # hardcoded release mode


find_package(Eigen3 REQUIRED)

include(ExternalProject)

set(LOCAL_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/local_install)
file(MAKE_DIRECTORY ${LOCAL_INSTALL_DIR})

if(SKIP_INSTALL_AD_TOOLS)
    # Add dummy targets
    add_custom_target(cppad-install)
    add_custom_target(cppad-cg-install)
else()
    # Download and populate cppad
    externalproject_add(
        cppad
        GIT_REPOSITORY       https://github.com/coin-or/CppAD.git
        GIT_TAG              master
        GIT_SHALLOW          ON
        CMAKE_ARGS           -Dcppad_cxx_flags="-std=c++11"
                             -DCMAKE_INSTALL_PREFIX:PATH=${LOCAL_INSTALL_DIR}
        EXCLUDE_FROM_ALL     TRUE
    )
    externalproject_add_steptargets(cppad install)

    # Download and populate cppad-cg
    externalproject_add(
        cppad-cg
        GIT_REPOSITORY       https://github.com/joaoleal/CppADCodeGen.git
        GIT_TAG              master
        GIT_SHALLOW          ON
        CMAKE_ARGS           -DCPPAD_HOME=${LOCAL_INSTALL_DIR}/include
                             -DCMAKE_INSTALL_PREFIX:PATH=${LOCAL_INSTALL_DIR}
        DEPENDS              cppad-install
        EXCLUDE_FROM_ALL     TRUE
    )
    externalproject_add_steptargets(cppad-cg install)

    set(SKIP_INSTALL_AD_TOOLS ON CACHE BOOL "Set to skip superbuild")
endif()


add_executable(run_speedtests src/run_speedtests.cpp)
add_dependencies(run_speedtests cppad-install cppad-cg-install)
target_include_directories(run_speedtests
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../         # autodiff
    ${LOCAL_INSTALL_DIR}/include            # cppad, cppad-cg
)
target_link_libraries(run_speedtests
    dl
    Eigen3::Eigen
    pthread
)
