#ifndef TESTS_HPP_
#define TESTS_HPP_

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/array_algebra.hpp>
#include <Eigen/Core>

#include <algorithm>
#include <random>


struct ConstantNtoN
{
  static constexpr char name[] = "ConstantNtoN";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>::Ones(x.size());
  }
};


struct CoefficientWise
{
  static constexpr char name[] = "CoefficientWise";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return (x + x.cwiseInverse()).array().sin().matrix();
  }
};


struct SumOfSquares
{
  static constexpr char name[] = "SumOfSquares";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 1, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return Eigen::Matrix<typename Derived::Scalar, 1, 1>(x.cwiseAbs2().sum());
  }
};


// Adept doesn't like this (operator*)
struct ODE
{
  static constexpr char name[] = "ODE";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    static constexpr std::size_t N = 10;

    using scalar_t = typename Derived::Scalar;
    using state_t = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>;

    auto x0 = x.eval();

    // TODO: move this to a constructor

    // create matrix with ones on super-diagonal
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime,
      Derived::RowsAtCompileTime> A(x.size(), x.size());
    A.setZero();
    A.template block(0, 1, x.size() - 1, x.size() - 1).diagonal().setOnes();

    boost::numeric::odeint::integrate_n_steps(
      boost::numeric::odeint::runge_kutta4<state_t, scalar_t, state_t, scalar_t,
      boost::numeric::odeint::vector_space_algebra>{},
      [&A](const state_t & x, state_t & dxdt, const scalar_t) {
        dxdt = A * x;
      },
      x0, scalar_t{0.}, scalar_t{0.01}, N
    );

    return x0;
  }
};


// Adept doesn't like this (operator*)
struct NeuralNet
{
  static constexpr char name[] = "NeuralNet";

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 1, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    static constexpr bool dynamic = Derived::RowsAtCompileTime == -1;
    static constexpr int n0 = Derived::RowsAtCompileTime;
    static constexpr int n1 = dynamic ? -1 : std::max<int>(Derived::RowsAtCompileTime / 2, 1);
    static constexpr int n2 = dynamic ? -1 : std::max<int>(Derived::RowsAtCompileTime / 4, 1);
    static constexpr int n3 = dynamic ? -1 : std::max<int>(Derived::RowsAtCompileTime / 8, 1);

    int n0_dyn = x.size();
    int n1_dyn = std::max<int>(n0_dyn / 2, 1);
    int n2_dyn = std::max<int>(n1_dyn / 2, 1);
    int n3_dyn = std::max<int>(n2_dyn / 2, 1);

    // TODO: move this to a constructor, template on fixed input size
    std::minstd_rand gen(101);  // fixed seed
    std::normal_distribution<double> dis(0, 1);

    Eigen::Matrix<typename Derived::Scalar, n1, n0> W1;
    W1 = Eigen::Matrix<typename Derived::Scalar, n1, n0>::NullaryExpr(
      n1_dyn, n0_dyn,
      [&]() {
        return static_cast<typename Derived::Scalar>(0.1 * dis(gen));
      });

    Eigen::Matrix<typename Derived::Scalar, n2, n1> W2;
    W2 = Eigen::Matrix<typename Derived::Scalar, n2, n1>::NullaryExpr(
      n2_dyn, n1_dyn,
      [&]() {
        return static_cast<typename Derived::Scalar>(0.1 * dis(gen));
      });

    Eigen::Matrix<typename Derived::Scalar, n3, n2> W3;
    W3 = Eigen::Matrix<typename Derived::Scalar, n3, n2>::NullaryExpr(
      n3_dyn, n2_dyn,
      [&]() {
        return static_cast<typename Derived::Scalar>(0.1 * dis(gen));
      });

    auto z1 = (W1 * x).eval();
    auto a1 = z1.array().tanh().matrix().eval();

    auto z2 = (W2 * a1).eval();
    auto a2 = z2.array().tanh().matrix().eval();

    auto z3 = (W3 * a2).eval();
    auto a3 = z3.array().tanh().matrix().eval();

    return Eigen::Matrix<typename Derived::Scalar, 1, 1>(a3.squaredNorm());
  }
};


#endif  // TESTS_HPP_
