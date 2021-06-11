#ifndef UPDATE_ALPHA_3_H
#define UPDATE_ALPHA_3_H

#include <RcppArmadillo.h>
#include <cmath>
#include "UpdatePi.h"


void updateAlpha3(const arma::vec& pi,
                  const double& b,
                  const arma::mat& Z,
                  const int& iter,
                  const int& tot_mcmc_iters,
                  const double& sigma_alpha_3,
                  arma::vec& alpha_3);


#endif
