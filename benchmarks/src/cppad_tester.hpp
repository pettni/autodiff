#include <cppad/cppad.hpp>

#include "test_interface.hpp"

template<typename T>
class CppADTester : public TestInterface<CppADTester<T>>
{
public:
  static constexpr char name[] = "CppAD";

  void setup(uint32_t dynamic_size)
  {
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ax(dynamic_size);
    ax.setOnes();

    CppAD::Independent(ax);
    Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> ay = T()(ax);
    f_ad = CppAD::ADFun<double>(ax, ay);

    f_ad.optimize();
  }

  Eigen::MatrixXd run(const Eigen::VectorXd & x)
  {
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> J(f_ad.Domain(), f_ad.Range());
    Eigen::Map<Eigen::MatrixXd>(J.data(), f_ad.Domain(), f_ad.Range()) = f_ad.Jacobian(x);
    return J;
  }

private:
  CppAD::ADFun<double> f_ad;
};
