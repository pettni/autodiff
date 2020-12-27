#include <Eigen/Core>

#include <autodiff/common/meta.hpp>

#include <iostream>

#include "autodiff_testers.hpp"
#include "cppadcg_tester.hpp"
#include "cppad_tester.hpp"

#include "tests.hpp"


template<template<typename> typename Tester, typename Test, uint32_t size>
void run_speetest()
{
  if constexpr (
    std::is_same_v<Tester<Test>, AutodiffRevStaticTester<BenchMark10>>||
    std::is_same_v<Tester<Test>, AutodiffRevDynamicTester<BenchMark10>>)
  {
    // test too slow for a single iter
    std::cout <<
      std::left <<
      std::setw(25) << Tester<Test>::name <<
      "TIMEOUT" <<
      std::endl;
    return;
  }
  Tester<Test> test;
  auto res = test.template test_speed<size>();
  std::cout <<
    std::left <<
    std::setw(25) << Tester<Test>::name <<
    std::right << std::setw(12) << res.setup_time.count() / res.setup_iter <<
    std::right << std::setw(12) << res.setup_iter <<
    std::right << std::setw(12) << res.calc_time.count() / res.calc_iter <<
    std::right << std::setw(12) << res.calc_iter <<
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
      (run_speetest<Tester, typename TestPack::template type<i>, 2>(), ...);
      std::cout << "=== TestPack<" << i << "> (n=5) ===" << std::endl;
      (run_speetest<Tester, typename TestPack::template type<i>, 5>(), ...);
      std::cout << "=== TestPack<" << i << "> (n=10) ===" << std::endl;
      (run_speetest<Tester, typename TestPack::template type<i>, 10>(), ...);
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
    AutodiffFwdStaticTester,
    AutodiffFwdDynamicTester,
    AutodiffRevStaticTester,
    AutodiffRevDynamicTester,
    CppADTester,
    CppADCGTester
  >();

  return 0;
}
