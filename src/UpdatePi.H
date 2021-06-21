#ifndef UPDATE_PI_H
#define UPDATE_PI_H

#include <RcppArmadillo.h>
#include <cmath>

void updatePi(const double& alpha,
               const arma::mat& Z,
               const int& iter,
               const int& tot_mcmc_iters,
               arma::mat& pi);

void updatePi_PM(const double& alpha_3,
                 const arma::mat& Z,
                 const arma::vec& c,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 const double& a_pi_PM,
                 arma::vec& pi_ph,
                 arma::mat& pi);

#endif

