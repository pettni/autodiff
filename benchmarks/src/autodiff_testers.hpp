#ifndef SRC__AUTODIFF_TESTERS_HPP_
#define SRC__AUTODIFF_TESTERS_HPP_


#include <autodiff/common/meta.hpp>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>

#include "test_interface.hpp"


template<typename T>
class AutodiffFwdStaticTester : public TestInterface<AutodiffFwdStaticTester<T>>
{
public:
  static constexpr char name[] = "AutodiffFwdStatic";

  template<uint32_t _nX>
  void setup()
  {}

  template<uint32_t _nX>
  Eigen::Matrix<double, _nX, _nX>
  run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::Matrix<autodiff::dual, _nX, 1> x_ad = x.template cast<autodiff::dual>();

    return autodiff::forward::jacobian(
      [this](const Eigen::Matrix<autodiff::dual, _nX, 1> & x)
      -> Eigen::Matrix<autodiff::dual, _nX, 1> {
        return f(x);
      },
      autodiff::forward::wrt(x_ad),
      autodiff::forward::at(x_ad)
    );
  }

private:
  T f{};
};


template<typename T>
class AutodiffFwdDynamicTester : public TestInterface<AutodiffFwdDynamicTester<T>>
{
public:
  static constexpr char name[] = "AutodiffFwdDynamic";

  template<uint32_t _nX>
  void setup()
  {}

  template<uint32_t _nX>
  Eigen::Matrix<double, _nX, _nX>
  run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::Matrix<autodiff::dual, -1, 1> x_ad = x.template cast<autodiff::dual>();

    return autodiff::forward::jacobian(
      [this](const Eigen::Matrix<autodiff::dual, -1, 1> & x)
      -> Eigen::Matrix<autodiff::dual, -1, 1> {
        return f(x);
      },
      autodiff::forward::wrt(x_ad),
      autodiff::forward::at(x_ad)
    );
  }

private:
  T f{};
};


template<typename T>
class AutodiffRevStaticTester : public TestInterface<AutodiffRevStaticTester<T>>
{
public:
  static constexpr char name[] = "AutodiffRevStatic";

  template<uint32_t _nX>
  void setup()
  {}

  template<uint32_t _nX>
  Eigen::Matrix<double, _nX, _nX>
  run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::Matrix<autodiff::var, _nX, 1> x_ad = x.template cast<autodiff::var>();
    Eigen::Matrix<autodiff::var, _nX, 1> F = f(x_ad);

    Eigen::Matrix<double, _nX, _nX> J;
    autodiff::detail::For<_nX>(
      [&J, &F, &x_ad](auto i) {
        J.row(i) = autodiff::reverse::gradient(F(i), x_ad);
      });
    return J;
  }

private:
  T f{};
};


template<typename T>
class AutodiffRevDynamicTester : public TestInterface<AutodiffRevDynamicTester<T>>
{
public:
  static constexpr char name[] = "AutodiffRevDynamic";

  template<uint32_t _nX>
  void setup()
  {}

  template<uint32_t _nX>
  Eigen::Matrix<double, _nX, _nX>
  run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::Matrix<autodiff::var, -1, 1> x_ad = x.template cast<autodiff::var>();
    Eigen::Matrix<autodiff::var, -1, 1> F = f(x_ad);

    Eigen::MatrixXd J(F.rows(), x.rows());
    for (auto i = 0u; i != F.rows(); ++i) {
      J.row(i) = autodiff::reverse::gradient(F(i), x_ad);
    }
    return J;
  }

private:
  T f{};
};

#endif  // SRC__AUTODIFF_TESTERS_HPP_
