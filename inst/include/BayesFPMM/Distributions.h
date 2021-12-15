#ifndef BayesFPMM_DISTRIBUTIONS_H
#define BayesFPMM_DISTRIBUTIONS_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFPMM {
// Calculates log gamma of a double
//
// @name logGamma
// @param x Double
// @return log(x)
inline double logGamma(double x){
  return log(tgamma(x));
}

// Generates a random sample from the Dirichlet Distribution
//
// @name rdirichlet
// @param alpha Vector containing concentration parameters
// @returns distribution Vector containing the random sample
inline arma::vec rdirichlet(arma::vec alpha){
  arma::vec distribution(alpha.n_elem, arma::fill::zeros);

  double sum_term = 0;

  for (int j = 0; j < alpha.n_elem; ++j) {
    double gam = R::rgamma(alpha[j],1.0);
    distribution(j) = gam;
    sum_term += gam;
  }

  for (int j = 0; j < alpha.n_elem; ++j) {
    distribution(j) = distribution(j) / sum_term;
  }

  return distribution;
}

// Calculates the log of B(a) function used in the dirichlet distribution
//
// @name calc_lB
// @param alpha Vector containing input to the function
// @return log_B Double containing the log of the output of the B function
inline double calc_lB(const arma::vec& alpha){
  double log_B = 0;

  for(int i=0; i < alpha.n_elem; i++){
    log_B = log_B + std::lgamma(alpha(i));
  }
  log_B = log_B - std::lgamma(arma::accu(alpha));

  return log_B;
}

}

#endif
