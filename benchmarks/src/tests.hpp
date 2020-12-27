#include <Eigen/Core>


struct BenchMark1
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    return Derived::Ones(x.size());
  }
};


struct BenchMark2
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    int n = x.size();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = i;
    }
    return res;
  }
};


struct BenchMark3
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    const auto n = x.size();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = -i / (x[i] * x[i]);
    }
    return res;
  }
};


struct BenchMark4
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    using std::sqrt;
    const auto n = x.size();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = 1 + i / sqrt(x[i]);
    }
    return res;
  }
};


struct BenchMark5
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    using std::sqrt;
    const auto n = x.size();
    const auto fval = sqrt(x.cwiseAbs2().sum());
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = x[i] / fval;
    }
    return res;
  }
};


struct BenchMark6
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    const auto n = x.size();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = 1.0 / x[i];
    }
    return res;
  }
};


struct BenchMark7
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    using std::log;
    const auto n = x.size();
    const auto xsum = x.sum();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = log(x[i] / xsum);
    }
    return res;
  }
};


struct BenchMark8
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    using std::sin;
    using std::cos;
    const auto n = x.size();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      const auto sin_xi = sin(x[i]);
      const auto cos_xi = cos(x[i]);
      res[i] = cos_xi * cos_xi - sin_xi * sin_xi;
    }
    return res;
  }
};


struct BenchMark9
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    using std::exp;
    const auto n = x.size();
    const typename Derived::Scalar fval = x.array().exp().matrix().sum();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = typename Derived::Scalar(i) / x[i] * (1. + 1. / x[i]) * exp(x[i] - fval);
    }
    return res;
  }
};


struct BenchMark10
{
  template<typename Derived>
  decltype(auto) operator()(const Eigen::MatrixBase<Derived>&x)
  {
    using std::log;
    const auto n = x.size();
    Derived res(n);
    for (auto i = 0; i < n; ++i) {
      res[i] = 2. + 2. * x[i] + 3. * x[i] * x[i] - 1. / (x[i] * x[i]) - 2. / (x[i] * x[i] * x[i]) -
        3. /
        (x[i] * x[i] * x[i] * x[i]) + log(x[i]);
    }
    return res;
  }
};
