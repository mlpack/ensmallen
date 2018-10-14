using Optim

if length(ARGS) < 2
  print("need two arguments: dim, points")
  exit(1)
end

dim = parse(Int, ARGS[1])
points = parse(Int, ARGS[2])

# Linear regression Evaluate().
function f(theta, X, y)
    v = (y' - theta' * X)
    return sum(v .* v)
end

# Use Optim and LBFGS to optimize.
X = rand(dim, points)
y = rand(points)

for i in 1:points
  a = rand(1)
  X[1, i] += a[1]
  y[i] += a[1]
end

result = optimize(t -> f(t, X, y), rand(dim),
    LBFGS(), Optim.Options(x_tol = 0, f_tol = 0, g_tol = 0, iterations = 10); autodiff = :forward)
print(result)
print("\n")

@time optimize(t -> f(t, X, y), rand(dim),
    LBFGS(), Optim.Options(x_tol = 0, f_tol = 0, g_tol = 0, iterations = 10); autodiff = :forward)

