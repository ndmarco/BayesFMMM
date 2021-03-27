#ifndef UPDATE_DELTA_H
#define UPDATE_DELTA_H

#include <RcppArmadillo.h>
#include <cmath>

void updateDelta(const arma::cube& phi,
                 const arma::cube& gamma,
                 const arma::vec& a,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 arma::mat& delta);

#endif
