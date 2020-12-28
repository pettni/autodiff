#ifndef TESTER_UTILS_HPP
#define TESTER_UTILS_HPP

#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <utility>

#include "common.hpp"


template<typename Tester1, typename Tester2, typename Test, std::size_t _nX>
bool test_correctness()
{
  Eigen::Matrix<double, _nX, 1> x;
  x.setOnes();
  typename EigenFunctor<Test, decltype(x)>::JacobianType J1, J2;

  Test test;

  Tester1 tester1;
  Tester2 tester2;

  // setup
  tester1.setup([&test](const auto & x) {return test(x);}, x);
  tester2.setup([&test](const auto & x) {return test(x);}, x);

  // test
  for (size_t i = 0; i < 5; i++) {
    // ensure inputs are positive
    x = 2 * Eigen::Matrix<double, _nX, 1>::Ones() + Eigen::Matrix<double, _nX, 1>::Random();

    tester1.run([&test](const auto & x) {return test(x);}, x, J1);
    tester2.run([&test](const auto & x) {return test(x);}, x, J2);

    if (!J1.isApprox(J2, 1e-6)) {
      std::cerr << "Different jacobians detected!" << std::endl;
      std::cerr << "Jacobian from " << Tester1::name << std::endl;
      std::cerr << J1 << std::endl;
      std::cerr << "Jacobian from " << Tester2::name << std::endl;
      std::cerr << J2 << std::endl;
      return false;
    }
  }

  return true;
}


struct SpeedResult
{
  bool setup_timeout, calc_timeout;
  uint64_t setup_iter{}, calc_iter{};
  std::chrono::nanoseconds setup_time{}, calc_time{};
};


template<typename Tester, typename Test, std::size_t _nX>
SpeedResult test_speed()
{
  SpeedResult res{};

  Tester tester;
  Test test;

  std::atomic<bool> canceled = false;
  std::promise<std::pair<std::size_t, std::chrono::nanoseconds>> setup_promise;
  auto setup_ftr = setup_promise.get_future();

  std::thread setup_thr(
    [&tester, &test, &canceled, &setup_promise]() {
      Eigen::Matrix<double, _nX, 1> x;
      x.setOnes();

      std::size_t cntr = 0;
      const auto beg = std::chrono::high_resolution_clock::now();
      while (!canceled) {
        tester.template setup([&test](const auto & x) {return test(x);}, x);
        ++cntr;
      }
      const auto end = std::chrono::high_resolution_clock::now();

      setup_promise.set_value(std::make_pair(cntr, end - beg));
    });

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  canceled.store(true);

  if (setup_ftr.wait_for(std::chrono::milliseconds(1000)) == std::future_status::ready) {
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
  std::thread calc_thr([&test, &tester, &canceled, &calc_promise]() {
      Eigen::Matrix<double, _nX, 1> x;
      x.setOnes();
      Eigen::Matrix<double, _nX, _nX> J;

      std::size_t cntr = 0;
      const auto beg = std::chrono::high_resolution_clock::now();
      while (!canceled) {
        tester.template run([&test](const auto & x) {return test(x);}, x, J);
        ++cntr;
      }
      const auto end = std::chrono::high_resolution_clock::now();

      calc_promise.set_value(std::make_pair(cntr, end - beg));
    });

  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  canceled.store(true);

  if (calc_ftr.wait_for(std::chrono::milliseconds(1000)) == std::future_status::ready) {
    calc_thr.join();
    std::tie(res.calc_iter, res.calc_time) = calc_ftr.get();
  } else {
    calc_thr.detach();    // forceful termination
    res.calc_timeout = true;
  }

  return res;
}

#endif  // TESTER_UTILS_HPP
