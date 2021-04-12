#ifndef CALCULATE__LIKELIHOOD_H
#define CALCULATE__LIKELIHOOD_H

#include <RcppArmadillo.h>
#include <cmath>

double calcLikelihood(const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& y_star,
                      const arma::field<arma::mat>& B_obs,
                      const arma::field<arma::mat>& B_star,
                      const arma::mat& nu,
                      const arma::cube& Phi,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const int& iter,
                      const double& sigma);

#endif
