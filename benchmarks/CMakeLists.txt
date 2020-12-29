cmake_minimum_required(VERSION 3.11)
project(autodiff_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# hardcoded release mode
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(Eigen3 REQUIRED)

include(ExternalProject)

set(LOCAL_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/local_install)
file(MAKE_DIRECTORY ${LOCAL_INSTALL_DIR})


if(SKIP_INSTALL_AD_TOOLS)
    # Add dummy targets
    add_custom_target(adept-external-install)
    add_custom_target(adolc-external-install)
    add_custom_target(cppad-external-install)
    add_custom_target(cppad-cg-external-install)
else()
    set(CONFIGURE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX11_STANDARD_COMPILE_OPTION}")

    # Download and install Adept
    externalproject_add(
        adept-external
        URL                  http://www.met.reading.ac.uk/clouds/adept/adept-2.0.8.tar.gz
        UPDATE_COMMAND       ""
        UPDATE_DISCONNECTED  ON
        # Need this patch to make adept work with Eigen (prevents gobbling constructor & ambiguous operator*)
        PATCH_COMMAND        patch -p0 -ruN -d <SOURCE_DIR> < ${CMAKE_CURRENT_SOURCE_DIR}/patch/adept_sfinae.patch
        CONFIGURE_COMMAND    ${CMAKE_COMMAND} -E env
                             CC=${CMAKE_C_COMPILER}
                             CFLAGS=${CMAKE_C_FLAGS_RELEASE}
                             CXX=${CMAKE_CXX_COMPILER}
                             CXXFLAGS=${CONFIGURE_CXX_FLAGS}
                             <SOURCE_DIR>/configure
                             --prefix=${LOCAL_INSTALL_DIR}
                             --disable-openmp
        BUILD_IN_SOURCE      ON
        EXCLUDE_FROM_ALL     ON
    )
    externalproject_add_steptargets(adept-external install)

    # Download and install ADOL-C
    externalproject_add(
        adolc-external
        GIT_REPOSITORY       git@github.com:coin-or/ADOL-C.git
        GIT_TAG              releases/2.7.2
        GIT_SHALLOW          ON
        GIT_CONFIG           advice.detachedHead=false
        UPDATE_COMMAND       ""
        UPDATE_DISCONNECTED  ON
        CONFIGURE_COMMAND    autoreconf -fi <SOURCE_DIR>
        COMMAND              ${CMAKE_COMMAND} -E env
                             CC=${CMAKE_C_COMPILER}
                             CFLAGS=${CMAKE_C_FLAGS_RELEASE}
                             CXX=${CMAKE_CXX_COMPILER}
                             CXXFLAGS=${CONFIGURE_CXX_FLAGS}
                             <SOURCE_DIR>/configure
                             --prefix=${LOCAL_INSTALL_DIR}
                             --includedir=${LOCAL_INSTALL_DIR}/include
                             --libdir=${LOCAL_INSTALL_DIR}/lib
                             --enable-traceless-refcounting  # hide warning in tapeless mode
        BUILD_IN_SOURCE      ON
        EXCLUDE_FROM_ALL     ON
    )
    externalproject_add_steptargets(adolc-external install)

    # Download and install CppAD
    externalproject_add(
        cppad-external
        GIT_REPOSITORY       https://github.com/coin-or/CppAD.git
        GIT_TAG              20200000.3
        GIT_SHALLOW          ON
        GIT_CONFIG           advice.detachedHead=false
        UPDATE_COMMAND       ""
        UPDATE_DISCONNECTED  ON
        CMAKE_ARGS           -Dcppad_cxx_flags=${CMAKE_CXX11_STANDARD_COMPILE_OPTION}
                             -DCMAKE_INSTALL_PREFIX:PATH=${LOCAL_INSTALL_DIR}
        EXCLUDE_FROM_ALL     ON
    )
    externalproject_add_steptargets(cppad-external install)

    # Download and install CppADCodeGen
    externalproject_add(
        cppad-cg-external
        GIT_REPOSITORY       https://github.com/joaoleal/CppADCodeGen.git
        GIT_TAG              v2.4.3
        GIT_SHALLOW          ON
        GIT_CONFIG           advice.detachedHead=false
        UPDATE_COMMAND       ""
        UPDATE_DISCONNECTED  ON
        CMAKE_ARGS           -DCPPAD_HOME=${LOCAL_INSTALL_DIR}/include
                             -DCMAKE_INSTALL_PREFIX:PATH=${LOCAL_INSTALL_DIR}
        DEPENDS              cppad-external-install
        EXCLUDE_FROM_ALL     ON
    )
    externalproject_add_steptargets(cppad-cg-external install)

    set(SKIP_INSTALL_AD_TOOLS ON CACHE BOOL "Set to skip superbuild")
endif()


add_executable(run_speedtests
    src/run_speedtests.cpp
)
add_dependencies(run_speedtests
    adept-external-install
    adolc-external-install
    cppad-external-install
    cppad-cg-external-install
)
target_compile_definitions(run_speedtests
PRIVATE
    ADOLC_NO_TAPING                    # use ADOL-C with no-taping forward mode (usually faster)
    CPPAD_FOR_TMB                      # required for cppad-cg on ODE test
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
    adept
    adolc
    dl
    Eigen3::Eigen
    pthread
    adolc
)
