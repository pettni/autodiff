#ifndef SRC__NUMERICAL_TESTER_HPP_
#define SRC__NUMERICAL_TESTER_HPP_


#include <unsupported/Eigen/NumericalDiff>

#include "test_interface.hpp"
#include "common.hpp"


template<typename T>
class NumericalTester : public TestInterface<NumericalTester<T>>
{
public:
  static constexpr char name[] = "Numerical";

  template<std::size_t _nX>
  void setup()
  {}

  template<std::size_t _nX>
  Eigen::Matrix<double, _nX, _nX> run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::NumericalDiff<EigenFunctor<_nX, _nX, T>> func;
    Eigen::Matrix<double, _nX, _nX> J;

    func.df(x, J);

    return J;
  }
};

#endif  // SRC__NUMERICAL_TESTER_HPP_
