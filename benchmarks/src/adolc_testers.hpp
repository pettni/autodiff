#ifndef SRC__ADOLC_EIGEN_TESTER_HPP_
#define SRC__ADOLC_EIGEN_TESTER_HPP_

#define ADOLC_NO_TAPING


#include "test_interface.hpp"

#ifdef ADOLC_NO_TAPING
#include <unsupported/Eigen/AdolcForward>
#include "common.hpp"

template<typename T>
class AdolcTester : public TestInterface<AdolcTester<T>>
{
public:
  static constexpr char name[] = "ADOL-C-TAPELESS";

  template<std::size_t _nX>
  void setup()
  {
    // can't get rid of annoying warning here...
    if (adtl::getNumDir() != _nX) {
      adtl::setNumDir(_nX);
    }
  }

  template<std::size_t _nX>
  Eigen::Matrix<double, _nX, _nX>
  run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::AdolcForwardJacobian<EigenFunctor<_nX, _nX, T>> func;
    Eigen::Matrix<double, _nX, 1> y;
    Eigen::Matrix<double, _nX, _nX> J;

    func(x, &y, &J);

    return J;
  }
};

#else  // ADOLC_NO_TAPING

#include <adolc/adolc.h>

template<typename T>
class AdolcTester : public TestInterface<AdolcTester<T>>
{
public:
  static constexpr char name[] = "ADOL-C";

  template<std::size_t _nX>
  void setup()
  {
    trace_on(0);

    double * x = new double[_nX];
    adouble * ax = new adouble[_nX];
    double * y = new double[_nX];
    adouble * ay = new adouble[_nX];

    for (size_t i = 0; i < _nX; i++) {
      x[i] = 1;
      ax[i] <<= x[i];
    }

    Eigen::Map<const Eigen::Matrix<adouble, _nX, 1>> ax_map(ax, _nX);
    Eigen::Map<Eigen::Matrix<adouble, _nX, 1>>(ay, _nX) = T()(ax_map);

    for (size_t i = 0; i < _nX; i++) {
      y[i] = 0;
      ay[i] >>= y[i];
    }

    trace_off();

    delete[] x;
    delete[] ax;
    delete[] y;
    delete[] ay;
  }

  template<std::size_t _nX>
  Eigen::Matrix<double, _nX, _nX>
  run(const Eigen::Matrix<double, _nX, 1> & x)
  {
    Eigen::Matrix<double, _nX, _nX, Eigen::RowMajor> J;

    std::array<double *, _nX> rows;
    for (size_t i = 0; i < _nX; i++) {
      rows[i] = J.data() + _nX * i;
    }

    jacobian(0, _nX, _nX, x.data(), rows.data());

    return J;
  }
};

#endif  // ADOLC_NO_TAPING


#endif  // SRC__ADOLC_EIGEN_TESTER_HPP_
