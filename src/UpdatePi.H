#ifndef UPDATE_PI_H
#define UPDATE_PI_H

#include <RcppArmadillo.h>
#include <cmath>

void update_pi(const double& alpha,
                 const int& iter,
                 const arma::mat& Z,
                 arma::mat& pi);

#endif
