% This is actually a GNU Octave file because I don't have access to the Global
% Optimization Toolkit.
pkg load optim;

args = argv();
dim = str2num(args{1, 1});
points = str2num(args{2, 1});

global x = rand(points, dim);
global y = rand(points, 1);

for i = 1:points,
  a = rand(1);
  x(i, 1) = x(i, 1) + a;
  y(i) = y(i) + a;
end

function [obj, grad] = lr(theta)
  global x;
  global y;

  v = (y - x * theta);
  obj = v' * v;
  grad = 2 * x' * v;
endfunction

% Force to run until convergence.  I guess maybe there is a JIT or something
% that might accelerate this, so let's let it "burn in" once.
theta0 = rand(dim, 1);
control = {10, 0}; % maxiters, verbosity
bfgsmin('lr', {theta0}, control);

tic;
bfgsmin('lr', {theta0}, control);
toc
