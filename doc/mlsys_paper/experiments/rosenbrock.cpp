/**
 * @file rosenbrock.cpp
 * @author Ryan Curtin
 *
 * A simple simulation that reports how long it takes to optimize the Rosenbrock
 * function using simulated annealing.
 */
#include <ensmallen.hpp>
#include <ensmallen_bits/problems/problems.hpp>

using namespace ens;
using namespace ens::test;

int main()
{
  // Use Armadillo's timing functionality to print how long it takes to
  // optimize.
  arma::wall_clock clock;

  RosenbrockFunction rf;
  ExponentialSchedule sched;
  // tolerance = 0 means it will run for the maximum number of iterations.
  SA<> s(sched, 100000, 10000, 1000, 100, 0.0);

  arma::mat parameters = rf.GetInitialPoint();
  clock.tic();

  s.Optimize(rf, parameters);

  const double time = clock.toc();

  std::cout << time << std::endl;
  std::cout << "Result (optimal 1, 1): " << parameters.t();
}
