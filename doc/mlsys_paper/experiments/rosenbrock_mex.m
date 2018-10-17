% This is actually a GNU Octave file because I don't have access to the Global
% Optimization Toolkit.
pkg load optim;

r = @(p) mex_rosenbrock(p);
% Force to run until convergence.  I guess maybe there is a JIT or something
% that might accelerate this, so let's let it "burn in" once.
pin = [-1.2; 1.0];
nonlin_min(r, pin, optimset('algorithm', 'siman', 'MaxIter', 100000,
'MaxFunEvals', 100000, 'TolX', 0, 'TolFun', 0));

tic;
nonlin_min(r, pin, optimset('algorithm', 'siman', 'MaxIter', 100000,
'MaxFunEvals', 100000, 'TolX', 0, 'TolFun', 0))
toc
