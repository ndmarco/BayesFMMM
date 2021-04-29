#ifndef UPDATE_Y_Star_H
#define UPDATE_Y_Star_H

#include <RcppArmadillo.h>
#include <cmath>

void updateYStar(const arma::field<arma::mat>& B_star,
                 const arma::mat& nu,
                 const arma::cube& Phi,
                 const arma::mat& Z,
                 const arma::mat& chi,
                 const double& sigma,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 arma::field<arma::mat>& y_star);

#endif
