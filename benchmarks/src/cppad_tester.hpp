#ifndef SRC__CPPAD_TESTER_HPP_
#define SRC__CPPAD_TESTER_HPP_


#include <cppad/cppad.hpp>

#include "test_interface.hpp"

template<typename T>
class CppADTester : public TestInterface<CppADTester<T>>
{
public:
  static constexpr char name[] = "CppAD";

  template<std::size_t _nX>
  void setup()
  {
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ax(_nX);
    ax.setOnes();

    CppAD::Independent(ax);
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ay = T()(ax);
    f_ad = CppAD::ADFun<double>(ax, ay);

    f_ad.optimize();
  }

  template<std::size_t _nX>
  Eigen::Matrix<double, _nX, _nX>
  run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::Matrix<double, _nX, _nX, Eigen::RowMajor> J;
    Eigen::VectorXd x_dyn = x;
    Eigen::Map<Eigen::VectorXd>(J.data(), _nX * _nX) = f_ad.Jacobian(x_dyn);
    return J;
  }

private:
  CppAD::ADFun<double> f_ad;
};

#endif  // SRC__CPPAD_TESTER_HPP_
