/*
TESTING RULES

* Task: compute dense jacobians of function Rn -> Rn
* Sizes known at compile time (benefits forward functions)
* No multithreading (benefits autodiff which does not support it)
* For tape methods employ any available optimization in the setup step

TESTS

* Support variable sized inputs, return same size output
* Can assume all inputs are positive
* No branching

TODO

* Add ADEPT: https://github.com/rjhogan/Adept-2
* Bold-face winner in each test: https://stackoverflow.com/questions/29997096/bold-output-in-c
* Simplify templating mess: no crtp, template setup() and run() on function instead of whole interface
* Test that mimics robotic dynamics
* Grab benchmarks from adept

*/

#ifndef TEST_INTERFACE_HPP_
#define TEST_INTERFACE_HPP_

#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <utility>


struct SpeedResult
{
  bool setup_timeout, calc_timeout;
  uint64_t setup_iter{}, calc_iter{};
  std::chrono::nanoseconds setup_time{}, calc_time{};
};


template<typename Derived>
class TestInterface
{
public:
  template<uint32_t size, typename OtherDerived>
  bool compare_with(TestInterface<OtherDerived> & other)
  {
    // setup
    static_cast<Derived &>(*this).template setup<size>();
    static_cast<OtherDerived &>(other).template setup<size>();

    // test
    for (size_t i = 0; i < 5; i++) {
      // ensure inputs are positive
      Eigen::Matrix<double, size, 1> X =
        2 * Eigen::VectorXd::Ones(size) + Eigen::VectorXd::Random(size);

      const auto J1 = static_cast<Derived &>(*this).template run<size>(X);
      const auto J2 = static_cast<OtherDerived &>(other).template run<size>(X);

      if (!J1.isApprox(J2, 1e-6)) {
        std::cout << "Different jacobians detected!" << std::endl;
        std::cout << "Jacobian from " << Derived::name << std::endl;
        std::cout << J1 << std::endl;
        std::cout << "Jacobian from " << OtherDerived::name << std::endl;
        std::cout << J2 << std::endl;
        return false;
      }
    }

    return true;
  }


  template<uint32_t _nX>
  SpeedResult test_speed()
  {
    SpeedResult res{};

    std::atomic<bool> canceled = false;
    std::promise<std::pair<std::size_t, std::chrono::nanoseconds>> setup_promise;
    auto setup_ftr = setup_promise.get_future();

    std::thread setup_thr(
      [this, &canceled, &setup_promise, &res]() {
        std::size_t cntr = 0;
        const auto beg = std::chrono::high_resolution_clock::now();
        while (!canceled) {
          static_cast<Derived &>(*this).template setup<_nX>();
          ++cntr;
        }
        const auto end = std::chrono::high_resolution_clock::now();

        setup_promise.set_value(std::make_pair(cntr, end - beg));
      });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    canceled.store(true);

    if (setup_ftr.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready) {
      setup_thr.join();
      std::tie(res.setup_iter, res.setup_time) = setup_ftr.get();
    } else {
      setup_thr.detach();
      res.setup_timeout = true;
      return res;
    }

    std::promise<std::pair<std::size_t, std::chrono::nanoseconds>> calc_promise;
    auto calc_ftr = calc_promise.get_future();
    canceled.store(false);
    std::thread calc_thr([this, &canceled, &calc_promise]() {
        Eigen::Matrix<double, _nX, 1> x;
        x.setOnes();

        std::size_t cntr = 0;
        const auto beg = std::chrono::high_resolution_clock::now();
        while (!canceled) {
          static_cast<Derived &>(*this).template run<_nX>(x);
          ++cntr;
        }
        const auto end = std::chrono::high_resolution_clock::now();

        calc_promise.set_value(std::make_pair(cntr, end - beg));
      });

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    canceled.store(true);

    if (calc_ftr.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready) {
      calc_thr.join();
      std::tie(res.calc_iter, res.calc_time) = calc_ftr.get();
    } else {
      calc_thr.detach();  // forceful termination
      res.calc_timeout = true;
    }

    return res;
  }
};

#endif  // TEST_INTERFACE_HPP_
