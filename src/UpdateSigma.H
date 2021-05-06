#ifndef UPDATE_SIGMA_H
#define UPDATE_SIGMA_H

#include <RcppArmadillo.h>
#include <cmath>

void updateSigma(const arma::field<arma::vec>& y_obs,
                 const arma::field<arma::mat>& y_star,
                 const arma::field<arma::mat>& B_obs,
                 const arma::field<arma::mat>& B_star,
                 const double alpha_0,
                 const double beta_0,
                 const arma::mat& nu,
                 const arma::cube& Phi,
                 const arma::mat& Z,
                 const arma::mat& chi,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 arma::vec& sigma);

void updateSigmaTempered(const double& beta_i,
                         const arma::field<arma::vec>& y_obs,
                         const arma::field<arma::mat>& y_star,
                         const arma::field<arma::mat>& B_obs,
                         const arma::field<arma::mat>& B_star,
                         const double alpha_0,
                         const double beta_0,
                         const arma::mat& nu,
                         const arma::cube& Phi,
                         const arma::mat& Z,
                         const arma::mat& chi,
                         const int& iter,
                         const int& tot_mcmc_iters,
                         arma::vec& sigma)
#endif
