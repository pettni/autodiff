#include <Eigen/Core>


struct BenchMark0
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>::Ones(x.size());
  }
};


struct BenchMark1
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    int n = x.size();
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1> res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = i;
    }
    return res;
  }
};


struct BenchMark2
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    const auto n = x.size();
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1> res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = -i / (x[i] * x[i]);
    }
    return res;
  }
};


struct BenchMark3
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    using std::sqrt;
    const auto n = x.size();
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1> res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = 1 + i / sqrt(x[i]);
    }
    return res;
  }
};


struct BenchMark4
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    using std::sqrt;
    return x / sqrt(x.cwiseAbs2().sum());
  }
};


struct BenchMark5
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    return x.cwiseInverse();
  }
};


struct BenchMark6
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    return x.array().log().matrix() / x.sum();
  }
};


struct BenchMark7
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    const auto sx = x.array().sin().eval();
    const auto cx = x.array().cos().eval();
    return (sx * sx - cx * cx).matrix();  // pow does not work for CppAD::cg::CG
  }
};


struct BenchMark8
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    using Scalar = typename Derived::Scalar;

    using std::exp;
    const auto n = x.size();
    const Scalar fval = x.array().exp().matrix().sum();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = Scalar(i) / x[i] * (1. + 1. / x[i]) * exp(x[i] - fval);
    }
    return res;
  }
};


struct BenchMark9
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    using Scalar = typename Derived::Scalar;
    return Scalar(2) * x +
           Scalar(3) * x.cwiseProduct(x) +
           Scalar(1) * x.cwiseProduct(x).cwiseInverse() +
           Scalar(2) * x.cwiseProduct(x).cwiseProduct(x).cwiseInverse() +
           Scalar(3) * x.cwiseProduct(x).cwiseProduct(x).cwiseProduct(x).cwiseInverse() +
           x.array().log().matrix();
  }
};


// Simulate numerical integration of an ODE using Euler
struct BenchMark10
{
  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x)
  {
    using Scalar = typename Derived::Scalar;
    using VecT = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1
    >;

    VecT x_cur = x;
    for (int i = 0; i != 100; ++i) {
      VecT dx = Scalar(-0.1) * x_cur + Scalar(0.5) * VecT::Ones(x.size());
      x_cur += dx;
    }
    return x_cur;
  }
};
