#ifndef CALCULATE_TT_ACCEPTANCE_H
#define CALCULATE_TT_ACCEPTANCE_H

#include <RcppArmadillo.h>
#include <cmath>

double calculatePZeta(const double& beta_i,
                      const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& y_star,
                      const arma::field<arma::mat>& B_obs,
                      const arma::field<arma::mat>& B_star,
                      const arma::mat nu,
                      const arma::cube& Phi,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const int& iter,
                      const double& sigma);

double CalculateTTAcceptance(const arma::vec& beta,
                             const arma::field<arma::vec>& y_obs,
                             const arma::field<arma::mat>& y_star,
                             const arma::field<arma::mat>& B_obs,
                             const arma::field<arma::mat>& B_star,
                             const arma::cube& nu,
                             const arma::field<arma::cube>& Phi,
                             const arma::cube& Z,
                             const arma::cube& chi,
                             const arma::vec& sigma);
#endif
