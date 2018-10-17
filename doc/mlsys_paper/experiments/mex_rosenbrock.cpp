#include "mex.h"
#include <stdio.h>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  if (nrhs == 0)
    return;
  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);

  double* s = mxGetPr(plhs[0]);

  const double* p = mxGetPr(prhs[0]);

  const double l = (p[1] - p[0] * p[0]);
  const double r = (1.0 - p[1]);
  *s = 100.0 * (l * l) + (r * r);

  return;
}
