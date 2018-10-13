% This is actually a GNU Octave file because I don't have access to the Global
% Optimization Toolkit.
pkg load optim;

rosen = @(p) 100 * (p(2) - p(1)^2)^2 + (1 - p(1))^2;

% Force to run until convergence.  I guess maybe there is a JIT or something
% that might accelerate this, so let's let it "burn in" once.
nonlin_min(rosen, [-1.2; 1.0], optimset('algorithm', 'siman', 'MaxIter', 100000,
'MaxFunEvals', 100000, 'TolX', 0, 'TolFun', 0));

tic;
nonlin_min(rosen, [-1.2; 1.0], optimset('algorithm', 'siman', 'MaxIter', 100000,
'MaxFunEvals', 100000, 'TolX', 0, 'TolFun', 0))
toc
