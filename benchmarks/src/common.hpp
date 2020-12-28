#ifndef SRC__COMMON_HPP_
#define SRC__COMMON_HPP_

#include <Eigen/Core>

template<std::size_t _nX, std::size_t _nY, typename T>
struct EigenFunctor : public T
{
  using Scalar = double;
  using InputType = Eigen::Matrix<double, _nX, 1>;
  using ValueType = Eigen::Matrix<double, _nY, 1>;
  static constexpr int InputsAtCompileTime = _nX;
  static constexpr int ValuesAtCompileTime = _nY;
  using JacobianType = Eigen::Matrix<double, _nY, _nX>;

  int values() const
  {
    return _nX;
  }

  template<typename Derived1, typename Derived2>
  void operator()(const Eigen::MatrixBase<Derived1> & x, Eigen::MatrixBase<Derived2> * y) const
  {
    *y = T::operator()(x);
  }

  template<typename Derived1, typename Derived2>
  void operator()(const Eigen::MatrixBase<Derived1> & x, Eigen::MatrixBase<Derived2> & y) const
  {
    y = T::operator()(x);
  }
};

#endif  // SRC__COMMON_HPP_
