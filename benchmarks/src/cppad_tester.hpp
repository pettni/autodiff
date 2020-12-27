#include <cppad/cppad.hpp>

#include "test_interface.hpp"

template<typename T>
class CppADTester : public TestInterface<CppADTester<T>, T>
{
public:
  static constexpr char name[] = "CppADTester";

  void setup(uint32_t dynamic_size)
  {
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ax(dynamic_size);
    ax.setOnes();

    CppAD::Independent(ax);
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ay = T()(ax);
    f_ad = CppAD::ADFun<double>(ax, ay);
  }

  Eigen::MatrixXd run(const Eigen::VectorXd & x)
  {
    return f_ad.Jacobian(x);
  }

private:
  CppAD::ADFun<double> f_ad;
};
