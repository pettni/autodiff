#ifndef TEST_INTERFACE_HPP_
#define TEST_INTERFACE_HPP_

#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <utility>

using namespace std::chrono_literals;


struct SpeedResult
{
  uint64_t setup_iter{}, calc_iter{};
  std::chrono::nanoseconds setup_time{}, calc_time{};
};


enum class TesterType { STATIC, DYNAMIC };


template<typename Derived, typename T>
class TestInterface
{
public:
  static constexpr TesterType type = TesterType::DYNAMIC;

  template<uint32_t size>
  SpeedResult test_speed()
  {
    SpeedResult res{};

    std::atomic<bool> canceled = false;
    std::pair<std::size_t, std::chrono::nanoseconds> setup_res, calc_res;

    std::thread thr_setup(
      [this, &canceled, &setup_res]() {
        std::size_t cntr = 0;
        const auto beg = std::chrono::high_resolution_clock::now();
        while (!canceled) {
          if constexpr (Derived::type == TesterType::DYNAMIC) {
            static_cast<Derived &>(*this).setup(size);
          }
          if constexpr (Derived::type == TesterType::STATIC) {
            static_cast<Derived &>(*this).template setup<size>();
          }
          ++cntr;
        }
        const auto end = std::chrono::high_resolution_clock::now();

        setup_res = std::make_pair(cntr, end - beg);
      });

    std::this_thread::sleep_for(500ms);
    canceled.store(true);
    thr_setup.join();

    canceled.store(false);
    std::thread calc_thr([this, &canceled, &calc_res]() {
        std::size_t cntr = 0;
        std::chrono::nanoseconds duration;

        if constexpr (Derived::type == TesterType::DYNAMIC) {
          Eigen::VectorXd x(size);
          x.setOnes();

          const auto beg = std::chrono::high_resolution_clock::now();
          while (!canceled) {
            static_cast<Derived &>(*this).run(x);
            ++cntr;
          }
          const auto end = std::chrono::high_resolution_clock::now();
          duration = end - beg;
        }

        if constexpr (Derived::type == TesterType::STATIC) {
          Eigen::Matrix<double, size, 1> x;
          x.setOnes();

          const auto beg = std::chrono::high_resolution_clock::now();
          while (!canceled) {
            static_cast<Derived &>(*this).template run<size>(x);
            ++cntr;
          }
          const auto end = std::chrono::high_resolution_clock::now();
          duration = end - beg;
        }

        calc_res = std::make_pair(cntr, duration);
      });

    std::this_thread::sleep_for(1s);
    canceled.store(true);
    calc_thr.join();

    res.setup_iter = setup_res.first;
    res.setup_time = setup_res.second;
    res.calc_iter = calc_res.first;
    res.calc_time = calc_res.second;

    return res;
  }
};

#endif  // TEST_INTERFACE_HPP_
