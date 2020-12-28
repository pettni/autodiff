#include <unsupported/Eigen/NumericalDiff>

#include <memory>

#include "test_interface.hpp"


template<typename T>
struct TFunctor : T
{
  explicit TFunctor(std::size_t size)
  : size_(size)
  {}

  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  static constexpr int InputsAtCompileTime = -1;
  static constexpr int ValuesAtCompileTime = -1;
  using JacobianType = Eigen::MatrixXd;

  int values() const
  {
    return size_;
  }

  void operator()(const InputType & x, ValueType & y) const
  {
    y = T::operator()(x);
  }

private:
  std::size_t size_;
};


template<typename T>
class NumericalTester : public TestInterface<NumericalTester<T>>
{
public:
  static constexpr char name[] = "Numerical";

  void setup(uint32_t dynamic_size)
  {
    nd = std::make_unique<Eigen::NumericalDiff<TFunctor<T>>>(
      TFunctor<T>(dynamic_size)
    );
  }

  Eigen::MatrixXd run(const Eigen::VectorXd & x)
  {
    Eigen::MatrixXd J(x.size(), x.size());

    nd->df(x, J);

    return J;
  }

private:
  std::unique_ptr<Eigen::NumericalDiff<TFunctor<T>>> nd;
};
