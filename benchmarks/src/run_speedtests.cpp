/*
PURPOSE

* Evaluate speed of computing

RULES

* Input/output sizes known at compile time (benefits forward methods)
* No multithreading
* For tape methods employ all available optimization in the recording step

TESTS

* Support variable sized inputs
* No branching (if/else) allowed

OTHER BENCHMARKS (CODE)

* Adept benchmarks: https://github.com/rjhogan/Adept-2/tree/master/benchmark
* Ceres benchmarks: https://github.com/ceres-solver/ceres-solver/tree/master/internal/ceres/autodiff_benchmarks
* "A Benchmark of Selected Algorithmic Differentiation Tools on Some Problems in Computer Vision and Machine Learning"
  - Paper: https://arxiv.org/abs/1807.10129
  - Code: https://github.com/microsoft/ADBench
  - Tools: Adept, ADOL-C, Ceres
* "Automatic Differentiation of Rigid Body Dynamics for Optimal Control and Estimation"
  - Paper: https://arxiv.org/abs/1709.03799
  - Tools: CppAD, CppAD-CG

TODOS

* Add option for dynamic-sized tests

AD TOOLS NOT IN BENCHMARK

* boost::math::differentiation     (Seems incompatible with array math)
* Enzime: https://enzyme.mit.edu/  (Clang compiler only)
* Fadbad: http://www.fadbad.com    (Seems unmaintained + Sacado claims to be faster successor)
* dco/c++                          (Non-free license)

*/

#include <Eigen/Core>

#include <autodiff/common/meta.hpp>

#include <exception>
#include <iomanip>
#include <iostream>

#include "adept_tester.hpp"
#include "adolc_testers.hpp"
#include "autodiff_testers.hpp"
#include "ceres_tester.hpp"
#include "cppadcg_tester.hpp"  // must be before cppad_tester
#include "cppad_tester.hpp"
#include "numerical_tester.hpp"
#include "sacado_tester.hpp"

#include "tests.hpp"
#include "tests_manif.hpp"

#include "test_utils.hpp"

template<typename Tester, typename Test>
void run_speedtest()
{
  auto res = test_speed<Tester, Test>();

  if (res.calc_timeout || res.setup_timeout) {
    std::cerr <<
      std::left << std::setw(20) << Tester::name <<
      std::left << std::setw(20) << Test::name <<
      "TIMEOUT (DETACHED)" <<
      std::endl;
    return;
  }

  // check if error occured
  if (!res.exception.empty()) {
    std::cerr <<
      std::left << std::setw(20) << Tester::name <<
      std::left << std::setw(20) << Test::name <<
      "EXCEPTION: " << res.exception <<
      std::endl;
    return;
  }

  // compare with numerical
  if (!test_correctness<Tester, NumericalTester, Test>()) {
    std::cerr <<
      std::left << std::setw(20) << Tester::name <<
      std::left << std::setw(20) << Test::name <<
      "CORRECTNESS ERROR" <<
      std::endl;
  }

  std::cout <<
    std::left << std::setw(20) << Tester::name <<
    std::left << std::setw(20) << Test::name <<
    std::left << std::setw(5) << Test::N <<
    std::setprecision(2) <<
    std::right << std::setw(10) << std::scientific <<
    static_cast<double>(res.setup_time.count()) / res.setup_iter <<
    std::right << std::setw(10) << std::scientific <<
    static_cast<double>(res.calc_time.count()) / res.calc_iter <<
    std::endl;
}


template<typename ... T>
struct TypePack
{
  static constexpr std::size_t size = sizeof...(T);

  template<std::size_t Idx>
  using type = std::tuple_element_t<Idx, std::tuple<T...>>;
};


template<typename TesterPack, typename TestPack>
void run_tests()
{
  autodiff::detail::For<TestPack::size>(
    [](auto test_i) {
      autodiff::detail::For<TesterPack::size>(
        [ = ](auto tester_i) {
          using Tester = typename TesterPack::template type<tester_i>;
          using Test = typename TestPack::template type<test_i>;
          run_speedtest<Tester, Test>();
        });
    });
}


int main()
{
  // these can run all tests (but do not necessarily succeed)
  run_tests<TypePack<
      AdeptTester,
      AdolcTester,
      AutodiffFwdTester,
      AutodiffRevTester,
      CppADTester,
      CppADCGTester,
      NumericalTester,
      SacadoTester
    >,
    TypePack<
      ConstantNtoN<3>,
      CoefficientWise<3>,
      SumOfSquares<3>,
      ODE<3>,
      NeuralNet<3>,
      ReprojectionError<3>,
      Manipulator<3>,
      SE3ODE<3>
    >
  >();

  // Ceres can't handle all tests
  run_tests<TypePack<
      CeresTester
    >,
    TypePack<
      ConstantNtoN<3>,
      CoefficientWise<3>,
      SumOfSquares<3>,
      // ODE<3>,               // double -> Jet not defined (required in boost)
      NeuralNet<3>,
      ReprojectionError<3>,
      Manipulator<3>
      // SE3ODE<3>      // double -> Jet not defined (required in boost)
    >
  >();


  // // Autodiff-rev can't run all tests
  // run_tests<TypePack<
  //     AutodiffRevTester
  //   >,
  //   TypePack<
  //     ConstantNtoN<3>,
  //     CoefficientWise<3>,
  //     SumOfSquares<3>,
  //     // ODE<3>,               // times out
  //     NeuralNet<3>,
  //     ReprojectionError<3>,
  //     Manipulator<3>
  //     // SE3ODE<3>      // times out
  //   >
  // >();

  return 0;
}
