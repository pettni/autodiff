/*
TESTING RULES

* Task: compute dense jacobians of function Rn -> Rn
* Sizes known at compile time (benefits forward functions)
* No multithreading (benefits autodiff which does not support it)
* For tape methods employ any available optimization in the setup step

TESTS

* Support variable sized inputs, return same size output
* Can assume all inputs are positive
* No branching

TODO

* Add ADEPT: https://github.com/rjhogan/Adept-2
* Bold-face winner in each test: https://stackoverflow.com/questions/29997096/bold-output-in-c
* Test that mimics robotic dynamics
* Grab benchmarks from adept

*/

#include <Eigen/Core>

#include <autodiff/common/meta.hpp>

#include <iomanip>
#include <iostream>

#include "adolc_testers.hpp"
#include "autodiff_testers.hpp"
#include "cppadcg_tester.hpp"
#include "cppad_tester.hpp"
#include "numerical_tester.hpp"
#include "tester_functions.hpp"

#include "tests.hpp"


template<typename Tester, typename Test, uint32_t size>
void run_speedtest()
{
  auto res = test_speed<Tester, Test, size>();

  if (res.calc_timeout || res.setup_timeout) {
    std::cout <<
      std::left << std::setw(20) << Tester::name <<
      std::right << std::setw(10) << "TIMEOUT (DETACHED)" <<
      std::endl;
    return;
  }

  // compare with numerical
  std::string name_str = Tester::name;
  if (!test_correctness<Tester, NumericalTester, Test, size>()) {
    name_str += " (ERROR)";
  }

  std::cout <<
    std::left << std::setw(20) << name_str <<
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
      std::cout << "=== " << TestPack::template type<test_i>::name << " ===" << std::endl;
      autodiff::detail::For<TesterPack::size>(
        [ = ](auto tester_i) {
          using Tester = typename TesterPack::template type<tester_i>;
          using Test = typename TestPack::template type<test_i>;
          run_speedtest<Tester, Test, 2>();
        });
    });
}


int main()
{
  using TestPack = TypePack<
    // BenchMark0,
    // BenchMark1,
    // BenchMark2,
    // BenchMark3,
    BenchMark4,
    BenchMark5,
    // BenchMark6,
    // BenchMark7,
    // BenchMark8,
    // BenchMark9,
    BenchMark10
  >;

  using TesterPack = TypePack<
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
