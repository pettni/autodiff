#ifndef TESTS_HPP_
#define TESTS_HPP_

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/array_algebra.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <random>


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


/**
 * Camera reprojection error for N points
 *
 * f(x)(2*i, 2*i+1) = ( x_C_i - proj(CM * (P_CW * exp(x)) * x_W_i) ) .^ 2
 *
 * where - x_C_i the i:th 2d pixel point
 *       - x_W_i the i:th 3d world point
 *       - P_CW a nominal camera pose
 *       - x is a tangent space element defining an incremental pose
 *
 * WARNING: DO NOT USE IN IMPORTANT CODE, MATH IS UNTESTED AND FRAGILE
 * USE A LIBRARY LIKE SOPHUS (https://github.com/stonier/sophus) OR
 * MANIF (https://github.com/artivis/manif) INSTEAD
 */
template<std::size_t _N>
struct ReprojectionError
{
  static constexpr char name[] = "Reprojection";
  static constexpr std::size_t N = _N;
  static constexpr std::size_t InputSize = 6;  //

  ReprojectionError()
  {
    // nominal pose
    t_nom = Eigen::Vector3d{0.1, -0.3, 0.2};
    q_nom.setIdentity();

    // camera matrix
    CM.setZero();
    CM(0, 0) = 700;  // fx
    CM(1, 1) = 690;  // fy
    CM(0, 2) = 320;  // cx
    CM(1, 2) = 240;  // cy
    CM(2, 2) = 1;

    // generate random data
    std::minstd_rand gen(101);  // fixed seed
    std::normal_distribution<double> dis(0, 1);
    auto gen_fcn = [&]() {return dis(gen);};

    for (std::size_t i = 0; i != N; ++i) {
      pts_world[i] = Eigen::Vector3d{0, 0, 3} + Eigen::Vector3d::NullaryExpr(gen_fcn);
      Eigen::Vector3d proj = CM * (t_nom + q_nom * pts_world[i]);
      pts_image[i] = proj.template head<2>() / proj(2);
    }
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2 * N, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    using Scalar = typename Derived::Scalar;
    using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Quat = Eigen::Quaternion<Scalar>;

    using std::sqrt, std::sin, std::cos;

    // Input is interpreted as a relative pose P_diff = (q_diff, t_diff)
    const auto t_diff = (Scalar{0.01} * x.template head<3>()).eval();
    const auto q_diff = (Scalar{0.01} * x.template tail<3>()).eval();

    // P_exp = exp(P_diff)   via SE(3) exponential
    const Scalar w_norm = q_diff.norm();
    const Vec3 q_diff_n = q_diff.normalized();

    Mat3 what;
    what << 0, -q_diff_n(2), q_diff_n(1),
      q_diff_n(2), 0, -q_diff_n(0),
      -q_diff_n(1), q_diff_n(0), 0;

    const Mat3 J = Mat3::Identity() +
      ((w_norm - sin(w_norm)) / w_norm) * what * what +
      ((1 - cos(w_norm)) / w_norm) * what;

    Vec3 t_exp = J * t_diff;
    Quat q_exp(Eigen::AngleAxis<Scalar>(w_norm, q_diff_n));

    // P_act = P_nom * P_exp  via SE(3) composition
    Quat q_act = q_nom.template cast<Scalar>() * q_exp;
    Vec3 t_act = t_nom.template cast<Scalar>() + q_nom.template cast<Scalar>() * t_exp;

    auto CMc = CM.template cast<Scalar>().eval();

    // Transform world points to camera frame, re-project, square
    Eigen::Matrix<Scalar, 2 * N, 1> ret;
    for (std::size_t i = 0; i != N; ++i) {
      Vec3 proj = CMc * (t_act + q_act * pts_world[i].template cast<Scalar>());
      ret.template segment<2>(2 * i) =
        (proj.template head<2>() / proj(2) - pts_image[i].template cast<Scalar>()).cwiseAbs2();
    }

    return ret;
  }

private:
  Eigen::Matrix<double, 3, 3> CM;            // camera matrix
  Eigen::Matrix<double, 3, 1> t_nom;         // nominal camera position
  Eigen::Quaterniond q_nom;                  // nominal camera orientation
  std::array<Eigen::Vector3d, N> pts_world;  // points in world frame
  std::array<Eigen::Vector2d, N> pts_image;  // points in image plane

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

#endif  // TESTS_HPP_
