#ifndef TESTS_MANIF_HPP_
#define TESTS_MANIF_HPP_

#include <manif/SE3.h>

#include <random>
#include <vector>


// namespace manif
// {

// template<typename _Scalar>
// struct Constants<adept::Active<_Scalar>>
// {
//   static const adept::Active<_Scalar> eps;
//   static const adept::Active<_Scalar> eps_s;
// };

// template<typename _Scalar>
// const adept::Active<_Scalar>
// Constants<adept::Active<_Scalar>>::eps = 1e-10;

// template<typename _Scalar>
// const adept::Active<_Scalar>
// Constants<adept::Active<_Scalar>>::eps_s = 1e-15;


// // for ADOL-C
// template<>
// struct Constants<adtl::adouble>
// {
//   static const adtl::adouble eps;
//   static const adtl::adouble eps_s;
// };

// const adtl::adouble
// Constants<adtl::adouble>::eps = adtl::adouble(1e-10);

// const adtl::adouble
// Constants<adtl::adouble>::eps_s = adtl::adouble(1e-15);


// // for autodiff dual
// template<typename _Scalar>
// struct Constants<autodiff::forward::Dual<_Scalar, _Scalar>>
// {
//   static const autodiff::forward::Dual<_Scalar, _Scalar> eps;
//   static const autodiff::forward::Dual<_Scalar, _Scalar> eps_s;
// };

// template<typename _Scalar>
// const autodiff::forward::Dual<_Scalar, _Scalar>
// Constants<autodiff::forward::Dual<_Scalar, _Scalar>>::eps = autodiff::forward::Dual<_Scalar,
//     _Scalar>(1e-10);

// template<typename _Scalar>
// const autodiff::forward::Dual<_Scalar, _Scalar>
// Constants<autodiff::forward::Dual<_Scalar, _Scalar>>::eps_s = autodiff::forward::Dual<_Scalar,
//     _Scalar>(1e-15);


// // for autodiff var
// template<>
// struct Constants<autodiff::var>
// {
//   static const autodiff::var eps;
//   static const autodiff::var eps_s;
// };

// const autodiff::var
// Constants<autodiff::var>::eps = autodiff::var(1e-10);

// const autodiff::var
// Constants<autodiff::var>::eps_s = autodiff::var(1e-15);


// // for autodiff var
// template<typename _Scalar>
// struct Constants<CppAD::AD<_Scalar>>
// {
//   static const CppAD::AD<_Scalar> eps;
//   static const CppAD::AD<_Scalar> eps_s;
// };

// template<typename _Scalar>
// const CppAD::AD<_Scalar>
// Constants<CppAD::AD<_Scalar>>::eps = CppAD::AD<_Scalar>(1e-10);

// template<typename _Scalar>
// const CppAD::AD<_Scalar>
// Constants<CppAD::AD<_Scalar>>::eps_s = CppAD::AD<_Scalar>(1e-15);


// // for sacado
// template<typename _Scalar>
// struct Constants<Sacado::Fad::DFad<_Scalar>>
// {
//   static const Sacado::Fad::DFad<_Scalar> eps;
//   static const Sacado::Fad::DFad<_Scalar> eps_s;
// };

// template<typename _Scalar>
// const Sacado::Fad::DFad<_Scalar>
// Constants<Sacado::Fad::DFad<_Scalar>>::eps = Sacado::Fad::DFad<_Scalar>(1e-10);

// template<typename _Scalar>
// const Sacado::Fad::DFad<_Scalar>
// Constants<Sacado::Fad::DFad<_Scalar>>::eps_s = Sacado::Fad::DFad<_Scalar>(1e-15);

// } /* namespace manif */


struct lie_operations
{
  template<class Fac1 = double, class Fac2 = Fac1>
  struct scale_sum2
  {
    const Fac1 m_alpha1;
    const Fac2 m_alpha2;

    scale_sum2(Fac1 alpha1, Fac2 alpha2)
    : m_alpha1(alpha1), m_alpha2(alpha2)
    {
      if (alpha1 != 1) {
        std::cerr << "alpha1 != 1 not supported for Lie integration" << std::endl;
        exit(1);
      }
    }

    template<class T1, class T2, class T3>
    void operator()(T1 & t1, const T2 & t2, const T3 & t3) const
    {
      t1 = t2 + t3 * m_alpha2;
    }

    typedef void result_type;
  };

  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2>
  struct scale_sum3
  {
    const scale_sum2<Fac1, Fac2> ss2;
    const Fac3 m_alpha3;

    scale_sum3(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3)
    : ss2(alpha1, alpha2), m_alpha3(alpha3) {}

    template<class T1, class T2, class T3, class T4>
    void operator()(T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4) const
    {
      ss2(t1, t2, t3);
      t1 += t4 * m_alpha3;
    }

    typedef void result_type;
  };

  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3>
  struct scale_sum4
  {
    const scale_sum3<Fac1, Fac2, Fac3> ss3;
    const Fac4 m_alpha4;

    scale_sum4(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3, Fac4 alpha4)
    : ss3(alpha1, alpha2, alpha3), m_alpha4(alpha4) {}

    template<class T1, class T2, class T3, class T4, class T5>
    void operator()(T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5) const
    {
      ss3(t1, t2, t3, t4);
      t1 += t5 * m_alpha4;
    }

    typedef void result_type;
  };

  template<class Fac1 = double, class Fac2 = Fac1, class Fac3 = Fac2, class Fac4 = Fac3,
    class Fac5 = Fac4>
  struct scale_sum5
  {
    const scale_sum4<Fac1, Fac2, Fac3, Fac4> ss4;
    const Fac5 m_alpha5;

    scale_sum5(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3, Fac4 alpha4, Fac5 alpha5)
    : ss4(alpha1, alpha2, alpha3, alpha4), m_alpha5(alpha5) {}

    template<class T1, class T2, class T3, class T4, class T5, class T6>
    void operator()(
      T1 & t1, const T2 & t2, const T3 & t3, const T4 & t4, const T5 & t5,
      const T6 & t6) const
    {
      ss4(t1, t2, t3, t4, t5);
      t1 += t6 * m_alpha5;
    }

    typedef void result_type;
  };
};


/**
 * Camera reprojection error for N points
 *
 * f: R^6 -> R^2N
 *
 * f(x)(2*i, 2*i+1) = ( x_C_i - proj(CM * (P_CW * exp(x)) * x_W_i) ) .^ 2
 *
 * where - x_C_i the i:th 2d pixel point
 *       - x_W_i the i:th 3d world point
 *       - P_CW a nominal camera pose
 *       - x is a tangent space element defining an incremental pose
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
    P_CW_nom = manif::SE3d(
      Eigen::Vector3d{0.1, -0.3, 0.2},
      Eigen::Quaterniond::Identity()
    );

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
      Eigen::Vector3d proj = CM * (P_CW_nom.act(pts_world[i]));
      pts_image[i] = proj.template head<2>() / proj(2);
    }
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2 * N, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    using Scalar = typename Derived::Scalar;
    using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
    using Quat = Eigen::Quaternion<Scalar>;

    manif::SE3Tangent<Scalar> delta_C = Scalar{0.01} * x;
    manif::SE3<Scalar> P_CW = delta_C + P_CW_nom.template cast<Scalar>();

    auto CMc = CM.template cast<Scalar>().eval();

    // Transform world points to camera frame, re-project, square
    Eigen::Matrix<Scalar, 2 * N, 1> ret;
    for (std::size_t i = 0; i != N; ++i) {
      Vec3 proj = CMc * (P_CW.act(pts_world[i].template cast<Scalar>()));
      ret.template segment<2>(2 * i) =
        (proj.template head<2>() / proj(2) - pts_image[i].template cast<Scalar>()).cwiseAbs2();
    }
    return ret;
  }

private:
  Eigen::Matrix<double, 3, 3> CM;            // camera matrix
  manif::SE3d P_CW_nom{};                    // nominal camera pose
  std::array<Eigen::Vector3d, N> pts_world;  // points in world frame
  std::array<Eigen::Vector2d, N> pts_image;  // points in image plane

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


/**
 * Differentiate the end effector position in an N link robotic arm
 *
 * f: R^6 -> R^6
 */
template<std::size_t _N>
struct Manipulator
{
  static constexpr char name[] = "Manipulator";
  static constexpr std::size_t N = _N;
  static constexpr std::size_t InputSize = 6;

  Manipulator()
  {
    // generate random link positions
    std::minstd_rand gen(101);  // fixed seed
    std::uniform_real_distribution<double> dis(-1, 1);
    auto gen_fcn = [&]() {return dis(gen);};

    for (std::size_t i = 0; i != N; ++i) {
      link_pose.emplace_back(
        2 * dis(gen), 2 * dis(gen), 2 * dis(gen),
        M_PI_2 * dis(gen), M_PI_2 * dis(gen), M_PI_2 * dis(gen)
      );
    }
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 3, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    using Scalar = typename Derived::Scalar;
    manif::SE3<Scalar> P = manif::SE3Tangent<Scalar>(x).exp();

    for (std::size_t i = 0; i != N; ++i) {
      P *= link_pose[i].template cast<Scalar>();
    }

    return P.act(Eigen::Matrix<typename Derived::Scalar, 3, 1>::UnitX());
  }

private:
  std::vector<manif::SE3d> link_pose{};
};


/**
 * Integrate a system on SE(3) for N steps starting at exp(x)
 *
 * f: R^6 -> R^6
 */
template<std::size_t _N>
struct SE3Integrator
{
  static constexpr char name[] = "SE3Integrator";
  static constexpr std::size_t N = _N;
  static constexpr std::size_t InputSize = 6;

  SE3Integrator()
  {
    velocity = (Eigen::Matrix<double, 6, 1>() << 0.1, -0.2, 0.3, 0.1, -0.2, 0.3).finished();
    Pfinal = manif::SE3d::Identity() + manif::SE3Tangentd(velocity);
  }

  template<typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 6, 1>
  operator()(const Eigen::MatrixBase<Derived> & x) const
  {
    using scalar_t = typename Derived::Scalar;
    using state_t = manif::SE3<scalar_t>;
    using deriv_t = typename state_t::Tangent;

    const manif::SE3Tangent<scalar_t> vel_c = velocity.template cast<scalar_t>();

    // set initial pose
    manif::SE3<scalar_t> P = manif::SE3Tangent<scalar_t>(x).exp();

    boost::numeric::odeint::integrate_n_steps(
      boost::numeric::odeint::runge_kutta4<state_t, scalar_t, deriv_t, scalar_t,
      boost::numeric::odeint::vector_space_algebra, lie_operations>{},
      [&vel_c](const state_t & X, deriv_t & dXdt, const scalar_t) {
        dXdt = vel_c;
      },
      P, scalar_t{0.}, scalar_t{0.01}, N
    );

    const manif::SE3<scalar_t> Pfinal_c = Pfinal.template cast<scalar_t>();
    return (P - Pfinal_c).coeffs();
  }

private:
  manif::SE3Tangentd velocity;
  manif::SE3d Pfinal = manif::SE3d::Identity();

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif  // TESTS_MANIF_HPP_
