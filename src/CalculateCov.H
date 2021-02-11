#ifndef CALCULATE_COV_H
#define CALCULATE_COV_H

#include <RcppArmadillo.h>
#include <cmath>

void getCov(const arma::rowvec& Z,
            const arma::cube& Phi,
            const arma::mat& Rho,
            arma::mat& Cov);
#endif
