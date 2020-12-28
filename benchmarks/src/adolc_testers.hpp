#ifndef SRC__ADOLC_TESTERS_HPP_
#define SRC__ADOLC_TESTERS_HPP_

#define ADOLC_NO_TAPING

#include <utility>

#ifdef ADOLC_NO_TAPING

# include <unsupported/Eigen/AdolcForward>
# include "common.hpp"


class AdolcTester
{
public:
  static constexpr char name[] = "ADOL-C (Tapeless)";

  template<typename Func, typename Derived>
  void setup(Func &&, const Eigen::PlainObjectBase<Derived> & x)
  {
    if (adtl::getNumDir() != x.size()) {
      adtl::setNumDir(x.size());
    }
  }

  template<typename Func, typename Derived>
  void run(
    Func && f,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    Eigen::AdolcForwardJacobian func(EigenFunctor<Func, Derived>(std::forward<Func>(f)));
    typename EigenFunctor<Func, Derived>::ValueType y;
    func(x, &y, &J);
  }
};

#else  // ADOLC_NO_TAPING

# include <adolc/adolc.h>

class AdolcTester
{
public:
  static constexpr char name[] = "ADOL-C";

  template<typename Func, typename Derived>
  void setup(Func && f, const Eigen::PlainObjectBase<Derived> & x)
  {
    trace_on(0);

    adouble * ax = new adouble[x.size()];
    double * y = new double[x.size()];
    adouble * ay = new adouble[x.size()];

    for (size_t i = 0; i < x.size(); i++) {
      ax[i] <<= x(i);
    }

    Eigen::Map<const Eigen::Matrix<adouble, -1, 1>> ax_map(ax, x.size());
    Eigen::Map<Eigen::Matrix<adouble, -1, 1>>(ay, x.size()) = f(ax_map);

    for (size_t i = 0; i < x.size(); i++) {
      y[i] = 0;
      ay[i] >>= y[i];
    }

    trace_off();

    delete[] ax;
    delete[] y;
    delete[] ay;
  }

  template<typename Func, typename Derived>
  void run(
    Func &&,
    const Eigen::PlainObjectBase<Derived> & x,
    typename EigenFunctor<Func, Derived>::JacobianType & J)
  {
    double ** rows = new double *[x.size()];

    for (size_t i = 0; i < Derived::RowsAtCompileTime; i++) {
      rows[i] = J.data() + Derived::RowsAtCompileTime * i;
    }

    jacobian(0, Derived::RowsAtCompileTime, Derived::RowsAtCompileTime, x.data(), rows);

    delete[] rows;
  }
};

#endif  // ADOLC_NO_TAPING


#endif  // SRC__ADOLC_TESTERS_HPP_
