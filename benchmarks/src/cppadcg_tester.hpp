#include <cppad/cg.hpp>

#include <memory>

#include "test_interface.hpp"

template<typename T>
class CppADCGTester : public TestInterface<CppADCGTester<T>>
{
public:
  static constexpr char name[] = "CppADCG";

  void setup(uint32_t size)
  {
    Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1> ax(size);
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

  Eigen::MatrixXd
  run(const Eigen::VectorXd & x)
  {
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> J(model->Domain(), model->Range());
    Eigen::Map<Eigen::MatrixXd>(J.data(), model->Domain(), model->Range()) = model->Jacobian(x);
    return J;
  }

private:
  std::unique_ptr<CppAD::cg::DynamicLib<double>> dynamicLib;
  std::unique_ptr<CppAD::cg::GenericModel<double>> model;
};
