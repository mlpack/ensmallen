/**
 * @file igd.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Inverse Generational Distance (IGD) indicator.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_INDICATORS_IGD_HPP
#define ENSMALLEN_INDICATORS_IGD_HPP

class IGD
{
  double Indicate(arma::cube& front,
  				  arma::cube& referenceFront)
  {
  	double igd = 0;
  	for (size_t i = 0; i < referenceFront.n_slices; i++)
  	{
  	  double min = DBL_MAX;
  	  for (size_t j = 0; j < front.n_slices; j++)
  	  {
  	  	double dist = 0;
  	  	for (size_t k = 0; k < front.slice(j).n_rows; k++)
  	  	{
  	  	  double z = referenceFront(k, 0, i);
  	  	  double a = front(k, 0, j);
  	  	  dist += std::pow(std::max(z - a, 0), 2);
  	  	}
  	  	dist = std::sqrt(dist);
  	  	if (dist < min)
  	  	  min = dist;
  	  }
  	  igd += std::pow(min, p);
  	}
  	igd = std::pow(igd, 1 / p);
  	igd /= referenceFront.n_slices;
  }
};

#endif