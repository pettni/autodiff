#include <adolc/adolc.h>

#include "test_interface.hpp"

template<typename T>
class AdolcTester : public TestInterface<AdolcTester<T>>
{
public:
  static constexpr char name[] = "ADOL-C";

  void setup(uint32_t dynamic_size)
  {
    dynamic_size_ = dynamic_size;

    trace_on(0);

    double * x = new double[dynamic_size];
    adouble * ax = new adouble[dynamic_size];
    double * y = new double[dynamic_size];
    adouble * ay = new adouble[dynamic_size];

    for (size_t i = 0; i < dynamic_size; i++) {
      x[i] = 1;
      ax[i] <<= x[i];
    }

    Eigen::Map<const Eigen::Matrix<adouble, -1, 1>> ax_map(ax, dynamic_size);
    auto result = T()(ax_map);
    Eigen::Map<Eigen::Matrix<adouble, -1, 1>>(ay, dynamic_size) = result;

    for (size_t i = 0; i < dynamic_size; i++) {
      y[i] = 0;
      ay[i] >>= y[i];
    }
    trace_off();

    delete[] x;
    delete[] ax;
    delete[] y;
    delete[] ay;
  }

  Eigen::MatrixXd run(const Eigen::VectorXd & x)
  {
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> J(dynamic_size_, dynamic_size_);
    double ** rows = new double *[dynamic_size_];
    for (size_t i = 0; i < dynamic_size_; i++) {
      rows[i] = J.data() + dynamic_size_ * i;
    }

    jacobian(0, dynamic_size_, dynamic_size_, x.data(), rows);

    return J;
  }

private:
  uint32_t dynamic_size_;
};
