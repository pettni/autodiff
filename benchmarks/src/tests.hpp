#ifndef TESTS_HPP_
#define TESTS_HPP_

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/array_algebra.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <random>
#include <utility>


/**
 * Return a constant vector
 *
 * f: R^N -> R^N
 */
template<std::size_t _N>
struct ConstantNtoN
{
  static constexpr char name[] = "ConstantNtoN";
  static constexpr std::size_t N = _N;
  static constexpr std::size_t InputSize = N;

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>::Ones(x.size());
  }
};


/**
 * Apply a series of coefficient-wise operations
 *
 * f: R^N -> R^N
 */
template<std::size_t _N>
struct CoefficientWise
{
  static constexpr char name[] = "CoefficientWise";
  static constexpr std::size_t N = _N;
  static constexpr std::size_t InputSize = N;

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return (x + x.cwiseInverse()).array().sin().matrix();
  }
};


/**
 * Calculate the sum of squares of the input vector
 *
 * f: R^N -> R
 *
 * f(x) = \sum_i x(i)^2
 */
template<std::size_t _N>
struct SumOfSquares
{
  static constexpr char name[] = "SumOfSquares";
  static constexpr std::size_t N = _N;
  static constexpr std::size_t InputSize = N;

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 1, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    return Eigen::Matrix<typename Derived::Scalar, 1, 1>(x.cwiseAbs2().sum());
  }
};


/**
 * Integrate an N-order integrator for 100 steps using a runge-kutta scheme
 *
 * f: R^N -> R^N
 */
template<std::size_t _N>
struct ODE
{
  static constexpr char name[] = "ODE";
  static constexpr std::size_t N = _N;
  static constexpr std::size_t InputSize = N;

  ODE()
  {
    // matrix with ones on super-diagonal
    A_.setZero();
    A_.template block(0, 1, N - 1, N - 1).diagonal().setOnes();
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    using scalar_t = typename Derived::Scalar;
    using state_t = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1>;

    auto x0 = x.eval();
    const auto Ac = A_.template cast<scalar_t>().eval();

    boost::numeric::odeint::integrate_n_steps(
      boost::numeric::odeint::runge_kutta4<
        state_t, scalar_t, state_t, scalar_t, boost::numeric::odeint::vector_space_algebra
      >{},
      [&Ac](const state_t & x, state_t & dxdt, const scalar_t) {
        dxdt = Ac * x;
      },
      x0, scalar_t{0.}, scalar_t{0.01}, 100
    );

    return x0;
  }

private:
  Eigen::Matrix<double, N, N> A_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


/**
 * Three-layer fully connected neural network with one channel and tanh activations
 *
 * f: R^N -> R
 *
 */
template<std::size_t _N>
struct NeuralNet
{
  static constexpr char name[] = "NeuralNet";
  static constexpr std::size_t N = _N;
  static constexpr std::size_t InputSize = N;

  NeuralNet()
  {
    std::minstd_rand gen(101);  // fixed seed
    std::normal_distribution<double> dis(0, 1);
    auto gen_fcn = [&]() {return dis(gen);};

    W1 = Eigen::Matrix<double, n1, n0>::NullaryExpr(gen_fcn);
    W2 = Eigen::Matrix<double, n2, n1>::NullaryExpr(gen_fcn);
    W3 = Eigen::Matrix<double, n3, n2>::NullaryExpr(gen_fcn);
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 1, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    auto z1 = (W1.template cast<typename Derived::Scalar>() * x.normalized()).eval();
    auto a1 = z1.array().tanh().matrix().eval();

    auto z2 = (W2.template cast<typename Derived::Scalar>() * a1).eval();
    auto a2 = z2.array().tanh().matrix().eval();

    auto z3 = (W3.template cast<typename Derived::Scalar>() * a2).eval();
    auto a3 = z3.array().tanh().matrix().eval();

    return Eigen::Matrix<typename Derived::Scalar, 1, 1>(
      (a3 - Eigen::Matrix<typename Derived::Scalar, n3, 1>::Ones()).squaredNorm()
    );
  }

private:
  static constexpr std::size_t n0 = N;
  static constexpr std::size_t n1 = std::max<int>(1, n0 / 2);
  static constexpr std::size_t n2 = std::max<int>(1, n1 / 2);
  static constexpr std::size_t n3 = std::max<int>(1, n2 / 2);

  Eigen::Matrix<double, n1, n0> W1;
  Eigen::Matrix<double, n2, n1> W2;
  Eigen::Matrix<double, n3, n2> W3;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif  // TESTS_HPP_
