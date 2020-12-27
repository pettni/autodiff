#ifndef TEST_INTERFACE_HPP_
#define TEST_INTERFACE_HPP_

#include <Eigen/Core>

#include <chrono>

struct SpeedResult
{
  uint32_t setup_iter{}, calc_iter{};
  std::chrono::nanoseconds setup_time{}, calc_time{};
};


enum class TesterType { STATIC, DYNAMIC };


template<typename Derived, typename T>
class TestInterface
{
public:
  static constexpr uint32_t setup_iter = 100000;
  static constexpr uint32_t calc_iter = 100000;

  static constexpr TesterType type = TesterType::DYNAMIC;

  template<uint32_t size>
  SpeedResult test_speed(
    uint32_t setup_iter = 100,
    uint32_t calc_iter = 100
  )
  {
    SpeedResult res{};
    res.setup_iter = setup_iter;
    res.calc_iter = calc_iter;

    const auto setup_begin = std::chrono::high_resolution_clock::now();
    for (auto i = 0u; i != setup_iter; ++i) {
      if constexpr (Derived::type == TesterType::DYNAMIC) {
        static_cast<Derived &>(*this).setup(size);
      }
      if constexpr (Derived::type == TesterType::STATIC) {
        static_cast<Derived &>(*this).template setup<size>();
      }
    }
    const auto setup_end = std::chrono::high_resolution_clock::now();
    res.setup_time = (setup_end - setup_begin) / setup_iter;


    if constexpr (Derived::type == TesterType::DYNAMIC) {
      // dynamic input size
      Eigen::VectorXd x(size);
      x.setOnes();

      const auto calculate_begin = std::chrono::high_resolution_clock::now();
      for (auto i = 0u; i != calc_iter; ++i) {
        static_cast<Derived &>(*this).run(x);
      }
      const auto calculate_end = std::chrono::high_resolution_clock::now();
      res.calc_time = (calculate_end - calculate_begin) / calc_iter;
    }

    if constexpr (Derived::type == TesterType::STATIC) {
      // static input size
      Eigen::Matrix<double, size, 1> x;
      x.setOnes();

      const auto calculate_begin = std::chrono::high_resolution_clock::now();
      for (auto i = 0u; i != calc_iter; ++i) {
        static_cast<Derived &>(*this).template run<size>(x);
      }
      const auto calculate_end = std::chrono::high_resolution_clock::now();
      res.calc_time = (calculate_end - calculate_begin) / calc_iter;
    }

    return res;
  }


};

#endif  // TEST_INTERFACE_HPP_
