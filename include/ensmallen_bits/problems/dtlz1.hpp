#ifndef ENSMALLEN_PROBLEMS_DTLZ_ONE_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_DTLZ_ONE_FUNCTION_HPP

namespace ens {
namespace test {

template <typename MatType = arma::mat>
class DTLZ1
{
private:
  size_t numParetoPoints {100};
  size_t numObjectives {30};
  size_t numVariables {30};


public:
  DTLZ1(size_t numVariables = 30, size_t numObjectives = 30)
  : numVariables(numVariables), numObjectives(numObjectives), objective(*this)
  {

  }

  size_t NumObjectives() const {return numObjectives;}

  size_t NumVariables() const {return numVariables;}

  arma::Col<typename MatType::elem_type> GetInitialPoint() const
  {
    typedef typename MatType::elem_type ElemType;

    return arma::Col<ElemType>(numVariables, 1, arma::fill::zeros);
  }

  struct Objectives
  {
    Objectives(DTLZ1& dtlzClass) : dtlzClass(dtlzClass){}

    size_t GetNumObjectives()
    {
      return dtlzClass.numObjectives;
    }

    arma::Col<typename MatType::elem_type> Evaluate(const MatType& coords)
    {
      arma::Col<typename MatType::elem_type> objectives(dtlzClass.numObjectives);

      size_t k = dtlzClass.numVariables - dtlzClass.numObjectives + 1;

      double g = 0;
      for (size_t i = dtlzClass.numVariables - k; i < dtlzClass.numVariables; i++)
      {
        double x = coords(i, 0);
        g += (x - 0.5) * (x - 0.5) - std::cos(20 * M_PI * (x - 0.5));
      }

      g = 100 * (k + g);
      for (size_t i = 0; i < dtlzClass.numObjectives; i++)
      {
        objectives(i) = (1 + g) / 2;
        for (size_t j = 0; j < dtlzClass.numObjectives - (i + 1); j++)
        {
          objectives(i) *= coords(j, 0);
        }
        if (i != 0)
        {
          objectives(i) *= 1 - coords(dtlzClass.numObjectives - (i + 1), 0);
        }
      }
      return objectives;
    }

    DTLZ1& dtlzClass;
  };

  Objectives GetObjectives()
  {
    return objective;
  }

  arma::Col<typename MatType::elem_type> Evaluate(const MatType& coords)
  {
    //typedef typename MatType::elem_type ElemType;

    arma::Col<typename MatType::elem_type> objectives(numObjectives);

    size_t k = numVariables - numObjectives + 1;

    double g = 0;
    for (size_t i = numVariables - k; i < numVariables; i++)
    {
      double x = coords(i, 0);
      g += (x - 0.5) * (x - 0.5) - std::cos(20 * M_PI * (x - 0.5));
    }

    g = 100 * (k + g);
    for (size_t i = 0; i < numVariables; i++)
    {
      objectives[i] = (1 + g) / 2;
      for (size_t j = 0; j < numObjectives - (i + 1); j++)
      {
        objectives[i] *= coords(j, 0);
      }
      if (i != 0)
      {
        objectives[i] *= 1 - coords(numObjectives - (i + 1), 0);
      }
    }
    return objectives;
  }
  Objectives objective;
};
}
}

#endif
