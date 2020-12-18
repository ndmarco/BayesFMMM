#ifndef UPDATE_GAMMA_H
#define UPDATE_GAMMA_H

#include <RcppArmadillo.h>
#include <cmath>

void updateGamma(const double& nu,
                 const int iter,
                 const arma::mat& delta,
                 const arma::cube& phi,
                 arma::field<arma::cube>& gamma);
void updateGamma(const double& nu,
                 const int iter,
                 const arma::vec& delta,
                 const arma::mat& phi,
                 arma::field<arma::mat>& gamma);
#endif
