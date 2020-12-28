#ifndef SRC__CPPADCG_TESTER_HPP_
#define SRC__CPPADCG_TESTER_HPP_


#include <cppad/cg.hpp>

#include <memory>

#include "test_interface.hpp"

template<typename T>
class CppADCGTester : public TestInterface<CppADCGTester<T>>
{
public:
  static constexpr char name[] = "CppADCG";

  template<std::size_t _nX>
  void setup()
  {
    Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1> ax(_nX);
    ax.setOnes();

    CppAD::Independent(ax);
    Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1> ay = T()(ax);
    CppAD::ADFun<CppAD::cg::CG<double>> f(ax, ay);
    f.optimize();

    // create dynamic library
    CppAD::cg::ModelCSourceGen<double> cgen(f, "model_Test");
    cgen.setCreateJacobian(true);
    cgen.setCreateHessian(false);
    cgen.setCreateForwardZero(false);
    cgen.setCreateForwardOne(false);
    cgen.setCreateReverseOne(false);
    cgen.setCreateReverseTwo(false);
    CppAD::cg::ModelLibraryCSourceGen<double> libcgen(cgen);

    // compile source code
    CppAD::cg::DynamicModelLibraryProcessor<double> p(libcgen);
    CppAD::cg::GccCompiler<double> compiler;
    dynamicLib = p.createDynamicLibrary(compiler);
    model = dynamicLib->model("model_Test");
  }

  template<std::size_t _nX>
  Eigen::Matrix<double, _nX, _nX>
  run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::Matrix<double, _nX, _nX, Eigen::RowMajor> J;
    Eigen::VectorXd x_dyn = x;
    Eigen::Map<Eigen::VectorXd>(J.data(), _nX * _nX) = model->Jacobian(x_dyn);
    return J;
  }

private:
  std::unique_ptr<CppAD::cg::DynamicLib<double>> dynamicLib;
  std::unique_ptr<CppAD::cg::GenericModel<double>> model;
};

#endif  // SRC__CPPADCG_TESTER_HPP_
