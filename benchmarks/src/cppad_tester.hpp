#ifndef SRC__CPPAD_TESTER_HPP_
#define SRC__CPPAD_TESTER_HPP_

#include <cppad/cppad.hpp>

#include "common.hpp"


class CppADTester
{
public:
  static constexpr char name[] = "CppAD";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ax = x.template cast<CppAD::AD<double>>();

    CppAD::Independent(ax);
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ay = f(ax);
    f_ad = CppAD::ADFun<double>(ax, ay);

    f_ad.optimize();
  }

  template<typename Func, typename Derived>
  void run(
    Func &&,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    Eigen::VectorXd x_dyn = x;
    Eigen::Map<Eigen::VectorXd>(J.data(), J.size()) = f_ad.Jacobian(x_dyn);
    J.transposeInPlace();
  }

private:
  CppAD::ADFun<double> f_ad;
};

#endif  // SRC__CPPAD_TESTER_HPP_
