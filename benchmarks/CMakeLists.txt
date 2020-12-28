cmake_minimum_required(VERSION 3.11)
project(autodiff_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)  # hardcoded release mode

find_package(Eigen3 REQUIRED)

include(ExternalProject)

set(LOCAL_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/local_install)
file(MAKE_DIRECTORY ${LOCAL_INSTALL_DIR})

if(SKIP_INSTALL_AD_TOOLS)
    # Add dummy targets
    add_custom_target(adolc-external-install)
    add_custom_target(cppad-external-install)
    add_custom_target(cppad-cg-external-install)
else()
    # Download and install adol-c
    set(ADOLC_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX11_STANDARD_COMPILE_OPTION}")
    externalproject_add(
        adolc-external
        GIT_REPOSITORY       git@github.com:coin-or/ADOL-C.git
        GIT_TAG              releases/2.7.2
        GIT_SHALLOW          ON
        GIT_CONFIG           advice.detachedHead=false
        UPDATE_DISCONNECTED  ON
        CONFIGURE_COMMAND    autoreconf -fi <SOURCE_DIR>
        COMMAND              ${CMAKE_COMMAND} -E env
                             CC=${CMAKE_C_COMPILER}
                             CFLAGS=${CMAKE_C_FLAGS_RELEASE}
                             CXX=${CMAKE_CXX_COMPILER}
                             CXXFLAGS=${ADOLC_CXX_FLAGS}
                             <SOURCE_DIR>/configure
                             --prefix=${LOCAL_INSTALL_DIR}
                             --includedir=${LOCAL_INSTALL_DIR}/include
                             --libdir=${LOCAL_INSTALL_DIR}/lib
        BUILD_IN_SOURCE      ON
        EXCLUDE_FROM_ALL     ON
    )
    externalproject_add_steptargets(adolc-external install)

    # Download and install cppad
    externalproject_add(
        cppad-external
        GIT_REPOSITORY       https://github.com/coin-or/CppAD.git
        GIT_TAG              20200000.3
        GIT_SHALLOW          ON
        GIT_CONFIG           advice.detachedHead=false
        UPDATE_DISCONNECTED  ON
        CMAKE_ARGS           -Dcppad_cxx_flags=${CMAKE_CXX11_STANDARD_COMPILE_OPTION}
                             -DCMAKE_INSTALL_PREFIX:PATH=${LOCAL_INSTALL_DIR}
        EXCLUDE_FROM_ALL     ON
    )
    externalproject_add_steptargets(cppad-external install)

    # Download and install cppad-cg
    externalproject_add(
        cppad-cg-external
        GIT_REPOSITORY       https://github.com/joaoleal/CppADCodeGen.git
        GIT_TAG              v2.4.3
        GIT_SHALLOW          ON
        GIT_CONFIG           advice.detachedHead=false
        UPDATE_DISCONNECTED  ON
        CMAKE_ARGS           -DCPPAD_HOME=${LOCAL_INSTALL_DIR}/include
                             -DCMAKE_INSTALL_PREFIX:PATH=${LOCAL_INSTALL_DIR}
        DEPENDS              cppad-external-install
        EXCLUDE_FROM_ALL     ON
    )
    externalproject_add_steptargets(cppad-cg-external install)

    set(SKIP_INSTALL_AD_TOOLS ON CACHE BOOL "Set to skip superbuild")
endif()


add_executable(adolc_test
    src/adolc_test.cpp
)
add_dependencies(adolc_test
    adolc-external-install
)
target_include_directories(adolc_test
PRIVATE
    ${LOCAL_INSTALL_DIR}/include
)
target_link_directories(adolc_test
PRIVATE
    ${LOCAL_INSTALL_DIR}/lib
)
target_link_libraries(adolc_test
    adolc
)


add_executable(run_speedtests
    src/run_speedtests.cpp
)
add_dependencies(run_speedtests
    adolc-external-install
    cppad-external-install
    cppad-cg-external-install
)
target_include_directories(run_speedtests
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../    # autodiff
    ${LOCAL_INSTALL_DIR}/include       # other AD libs
)
target_link_directories(run_speedtests
PRIVATE
    ${LOCAL_INSTALL_DIR}/lib
)
target_link_libraries(run_speedtests
    dl
    Eigen3::Eigen
    pthread
    adolc
)
