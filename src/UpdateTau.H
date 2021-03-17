#ifndef UPDATE_TAU_H
#define UPDATE_TAU_H

#include <RcppArmadillo.h>
#include <cmath>

void updateTau(const double& alpha,
               const double& beta,
               const arma::mat& nu,
               const int& iter,
               arma::mat& P,
               arma::mat& tau);

#endif
