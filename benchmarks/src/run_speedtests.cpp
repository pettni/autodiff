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
  Tester<Test> test;
  auto res = test.template test_speed<size>(Tester<Test>::setup_iter, Tester<Test>::calc_iter);
  std::cout <<
    std::left <<
    std::setw(30) << Tester<Test>::name <<
    std::right <<
    std::setw(15) << res.setup_time.count() <<
    " " <<
    std::setw(15) << res.calc_time.count() <<
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
      std::cout << "=== TestPack<" << i << "> (n=3) ===" << std::endl;
      (run_speetest<Tester, typename TestPack::template type<i>, 3>(), ...);
      std::cout << "=== TestPack<" << i << "> (n=10) ===" << std::endl;
      (run_speetest<Tester, typename TestPack::template type<i>, 10>(), ...);
    });
}


int main()
{
  using TestPack = TypePack<
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
