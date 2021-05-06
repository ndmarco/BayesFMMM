#ifndef UPDATE_CHI_H
#define UPDATE_CHI_H

#include <RcppArmadillo.h>

void updateChi(const arma::field<arma::vec>& y_obs,
               const arma::field<arma::mat>& y_star,
               const arma::field<arma::mat>& B_obs,
               const arma::field<arma::mat>& B_star,
               const arma::cube& Phi,
               const arma::mat& nu,
               const arma::mat& Z,
               const double& sigma,
               const int& iter,
               const int& tot_mcmc_iters,
               arma::cube& chi);

void updateChiTempered(const double& beta_i,
                       const arma::field<arma::vec>& y_obs,
                       const arma::field<arma::mat>& y_star,
                       const arma::field<arma::mat>& B_obs,
                       const arma::field<arma::mat>& B_star,
                       const arma::cube& Phi,
                       const arma::mat& nu,
                       const arma::mat& Z,
                       const double& sigma,
                       const int& iter,
                       const int& tot_mcmc_iters,
                       arma::cube& chi,
                       double& LP_zeta);

#endif
