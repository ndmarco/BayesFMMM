#ifndef UPDATE_GAMMA_H
#define UPDATE_GAMMA_H

#include <RcppArmadillo.h>
#include <cmath>

void updateGamma(const double& nu_gamma,
                 const arma::vec& delta,
                 const arma::cube& phi,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 arma::field<arma::cube>& gamma);
#endif
