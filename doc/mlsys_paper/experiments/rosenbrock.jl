using Optim
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

result = optimize(rosenbrock, [-1.2, 1.0], SimulatedAnnealing(),
    Optim.Options(g_tol=0.0, f_calls_limit=100000, iterations=100000))

print(result)
print("\n\n")

@time optimize(rosenbrock, [-1.2, 1.0], SimulatedAnnealing(),
    Optim.Options(g_tol=0.0, f_calls_limit=100000, iterations=100000))
