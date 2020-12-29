#ifndef SRC__CPPADCG_TESTER_HPP_
#define SRC__CPPADCG_TESTER_HPP_

#include <cppad/cg.hpp>

#include <memory>

#include "common.hpp"


class CppADCGTester
{
public:
  static constexpr char name[] = "CppADCG";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1> ax =
      x.template cast<CppAD::AD<CppAD::cg::CG<double>>>().eval();

    CppAD::Independent(ax);
    Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1> ay = f(ax);
    CppAD::ADFun<CppAD::cg::CG<double>> adfun(ax, ay);

    adfun.optimize();

    // create dynamic library
    CppAD::cg::ModelCSourceGen<double> cgen(adfun, "model_Test");
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

  template<typename Func, typename Derived>
  void run(
    Func &&,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    Eigen::VectorXd x_dyn = x;
    auto j_ad = model->Jacobian(x_dyn);

    using JacTRow = typename EigenFunctor<Func, Derived>::JacobianTypeRow;
    J = Eigen::Map<JacTRow>(j_ad.data(), j_ad.size() / x.size(), x.size());
  }

private:
  std::unique_ptr<CppAD::cg::DynamicLib<double>> dynamicLib;
  std::unique_ptr<CppAD::cg::GenericModel<double>> model;
};

#endif  // SRC__CPPADCG_TESTER_HPP_
