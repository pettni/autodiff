/*
TESTING RULES

* Task: compute dense jacobians of function Rn -> Rm
* Sizes known at compile time (benefits forward functions)
* No multithreading (benefits autodiff which does not support it)
* For tape methods employ any available optimization in the setup step

TESTS

* Support variable sized inputs
* Can assume all inputs are positive
* No branching

TODO

* Figure out how to handle tests with different (fixed) sizes
* Refine the tests
   - Camera reprojection error            (fixed size)
   - Differentiate SE3 lie group dynamics (fixed size)
   - Integrate SE3 lie group dynamics     (vairable integration steps)
* Let tests have (empty) constructors

OTHER BENCHMARKS

* Adept benchmarks: https://github.com/rjhogan/Adept-2/tree/master/benchmark
* 2015 paper w/ code: https://github.com/microsoft/ADBench
  - Adept, ADOL-C, Ceres
* Robotics dynamics: https://arxiv.org/abs/1709.03799
  - CppAD, CppAD-CG

AD TOOLS NOT IN BENCHMARK

* boost::math::differentiation     (Seems incompatible with array math)
* Enzime: https://enzyme.mit.edu/  (Clang compiler only)
* Fadbad: http://www.fadbad.com    (Seems unmaintained)
* Sacado                           (Website broken)
* dco/c++                          (Non-free license)

*/

#include <Eigen/Core>

#include <autodiff/common/meta.hpp>

#include <iomanip>
#include <iostream>

#include "adept_tester.hpp"
#include "adolc_testers.hpp"
#include "autodiff_testers.hpp"
#include "cppadcg_tester.hpp"  // must be before cppad_tester
#include "cppad_tester.hpp"
#include "numerical_tester.hpp"
#include "test_utils.hpp"

#include "tests.hpp"


template<typename Tester, typename Test, uint32_t size>
void run_speedtest()
{
  auto res = test_speed<Tester, Test, size>();

  if (res.calc_timeout || res.setup_timeout) {
    std::cerr <<
      std::left << std::setw(20) << Tester::name <<
      std::right << std::setw(10) << "TIMEOUT (DETACHED)" <<
      std::endl;
    return;
  }

  // compare with numerical
  std::string name_str = Tester::name;
  if (!test_correctness<Tester, NumericalTester, Test, size>()) {
    std::cerr << Tester::name << " gave incorrect answer on " << Test::name << std::endl;
    return;
  }

  std::cout <<
    std::left << std::setw(20) << Tester::name <<
    std::left << std::setw(20) << Test::name <<
    std::left << std::setw(5) << size <<
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
          run_speedtest<Tester, Test, 3>();
          // run_speedtest<Tester, Test, 5>();
          // run_speedtest<Tester, Test, 10>();
        });
    });
}


int main()
{
  using TestPack = TypePack<
    ConstantNtoN,
    CoefficientWise,
    SumOfSquares,
    ODE,
    NeuralNet
  >;

  using TesterPack = TypePack<
    AdeptTester,
    AdolcTester,
    AutodiffFwdTester,
    AutodiffRevTester,
    CppADTester,
    CppADCGTester,
    NumericalTester
  >;

  run_tests<TesterPack, TestPack>();

  return 0;
}
