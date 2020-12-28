#ifndef SRC__TESTS_HPP_
#define SRC__TESTS_HPP_

#include <Eigen/Core>


struct BenchMark0
{
  static constexpr char name[] = "BenchMark0";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>::Ones(x.size());
  }
};


struct BenchMark1
{
  static constexpr char name[] = "BenchMark1";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
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
  static constexpr char name[] = "BenchMark2";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
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
  static constexpr char name[] = "BenchMark3";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
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
  static constexpr char name[] = "BenchMark4";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    using std::sqrt;
    return x / sqrt(x.cwiseAbs2().sum());
  }
};


struct BenchMark5
{
  static constexpr char name[] = "BenchMark5";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return x.cwiseInverse();
  }
};


struct BenchMark6
{
  static constexpr char name[] = "BenchMark6";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return x.array().log().matrix() / x.sum();
  }
};


struct BenchMark7
{
  static constexpr char name[] = "BenchMark7";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    const auto sx = x.array().sin().eval();
    const auto cx = x.array().cos().eval();
    return (sx * sx - cx * cx).matrix();  // pow does not work for CppAD::cg::CG
  }
};


struct BenchMark8
{
  static constexpr char name[] = "BenchMark8";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    using Scalar = typename Derived::Scalar;

    using std::exp;
    const auto n = x.size();
    const Scalar fval = x.array().exp().matrix().sum();
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1> res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = Scalar(i) / x[i] * (1. + 1. / x[i]) * exp(x[i] - fval);
    }
    return res;
  }
};


struct BenchMark9
{
  static constexpr char name[] = "BenchMark9";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
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
  static constexpr char name[] = "BenchMark10";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    using Scalar = typename Derived::Scalar;
    using VecT = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1
    >;

    VecT x_cur = x;
    for (int i = 0; i != 100; ++i) {
      VecT dx = Scalar(-0.1) * x_cur.sum() * VecT::Ones(x.size()) +
        Scalar(3) * VecT::Ones(x.size());
      x_cur += Scalar(0.01) * dx;
    }
    return x_cur;
  }
};

#endif  // SRC__TESTS_HPP_
