#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <Eigen/Core>

#include <utility>


template<typename _Func, typename _InputType>
struct EigenFunctor
{
  using Scalar = typename _InputType::Scalar;
  using InputType = _InputType;
  using ValueType = std::invoke_result_t<_Func, InputType>;

  static constexpr int InputsAtCompileTime = InputType::SizeAtCompileTime;
  static constexpr int ValuesAtCompileTime = ValueType::SizeAtCompileTime;

  using JacobianType = Eigen::Matrix<
    Scalar, ValueType::SizeAtCompileTime, InputType::SizeAtCompileTime
  >;

  using JacobianTypeRow = Eigen::Matrix<
    Scalar, ValueType::SizeAtCompileTime, InputType::SizeAtCompileTime, Eigen::RowMajor
  >;

  int values_ = ValueType::SizeAtCompileTime;  // must be changed for dynamic sizing

  explicit EigenFunctor(_Func && func)
  : func_(std::forward<_Func>(func))
  {}

  int values() const
  {
    return values_;
  }

  template<typename Derived1, typename Derived2>
  void operator()(const Eigen::MatrixBase<Derived1> & x, Eigen::MatrixBase<Derived2> * y) const
  {
    * y = func_(x);
  }

  template<typename Derived1, typename Derived2>
  void operator()(const Eigen::MatrixBase<Derived1> & x, Eigen::MatrixBase<Derived2> & y) const
  {
    y = func_(x);
  }

private:
  _Func func_;
};

#endif  // COMMON_HPP_
