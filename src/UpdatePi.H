#ifndef UPDATE_PI_H
#define UPDATE_PI_H

#include <RcppArmadillo.h>
#include <cmath>

void update_pi(const double& alpha,
               const arma::mat& Z,
               const int& iter,
               const int& tot_mcmc_iters,
               arma::mat& pi);

#endif
