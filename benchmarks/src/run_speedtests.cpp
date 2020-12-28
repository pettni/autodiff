#include <Eigen/Core>

#include <autodiff/common/meta.hpp>

#include <iomanip>
#include <iostream>

#include "adolc_testers.hpp"
#include "autodiff_testers.hpp"
#include "cppadcg_tester.hpp"
#include "cppad_tester.hpp"
#include "numerical_tester.hpp"

#include "tests.hpp"


template<template<typename> typename Tester, typename Test, uint32_t size>
void run_speedtest()
{
  if constexpr (
    std::is_same_v<Tester<Test>, AutodiffRevStaticTester<BenchMark10>>||
    std::is_same_v<Tester<Test>, AutodiffRevDynamicTester<BenchMark10>>)
  {
    // test too slow for a single iter
    std::cout <<
      std::left << std::setw(20) << Tester<Test>::name <<
      std::right << std::setw(12) << "TIMEOUT" <<
      std::endl;
    return;
  }
  Tester<Test> test;

  // run the test
  auto res = test.template test_speed<size>();

  // compare with numerical
  NumericalTester<Test> cmp;
  bool correct = test.template compare_with<size>(cmp);
  std::string name_str = Tester<Test>::name;
  if (!correct) {
    name_str += " (ERROR)";
  }

  std::cout <<
    std::left <<
    std::setw(20) << name_str <<
    std::setprecision(2) <<
    std::right << std::setw(10) << std::scientific << static_cast<double>(res.setup_time.count()) /
    res.setup_iter <<
    std::right << std::setw(10) << std::scientific << static_cast<double>(res.calc_time.count()) /
    res.calc_iter <<
    std::endl;
}


template<typename ... T>
struct TypePack
{
  static constexpr std::size_t size = sizeof...(T);

  template<std::size_t Idx>
  using type = std::tuple_element_t<Idx, std::tuple<T...>>;
};


template<typename TestPack, template<typename> typename ... Tester>
void run_tests()
{
  autodiff::detail::For<TestPack::size>(
    [](auto i) {
      std::cout << "=== TestPack<" << i << "> (n=2) ===" << std::endl;
      (run_speedtest<Tester, typename TestPack::template type<i>, 2>(), ...);
      std::cout << "=== TestPack<" << i << "> (n=5) ===" << std::endl;
      (run_speedtest<Tester, typename TestPack::template type<i>, 5>(), ...);
      std::cout << "=== TestPack<" << i << "> (n=10) ===" << std::endl;
      (run_speedtest<Tester, typename TestPack::template type<i>, 10>(), ...);
    });
}


int main()
{
  using TestPack = TypePack<
    BenchMark0,
    BenchMark1,
    BenchMark2,
    BenchMark3,
    BenchMark4,
    BenchMark5,
    BenchMark6,
    BenchMark7,
    BenchMark8,
    BenchMark9,
    BenchMark10
  >;

  run_tests<
    TestPack,
    AdolcTester,
    AutodiffFwdStaticTester,
    AutodiffFwdDynamicTester,
    AutodiffRevStaticTester,
    AutodiffRevDynamicTester,
    CppADTester,
    CppADCGTester,
    NumericalTester
  >();

  return 0;
}
