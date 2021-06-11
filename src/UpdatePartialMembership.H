#ifndef UPDATE_PARTIAL_MEMBERSHIP_H
#define UPDATE_PARTIAL_MEMBERSHIP_H

#include <RcppArmadillo.h>
#include <cmath>

void updateZ_PM(const arma::field<arma::vec>& y_obs,
                const arma::field<arma::mat>& y_star,
                const arma::field<arma::mat>& B_obs,
                const arma::field<arma::mat>& B_star,
                const arma::cube& Phi,
                const arma::mat& nu,
                const arma::mat& chi,
                const arma::vec& pi,
                const double& sigma_sq,
                const double& rho,
                const int& iter,
                const int& tot_mcmc_iters,
                const double& alpha_3,
                const double& sigma_Z,
                arma::cube& Z_tilde,
                arma::mat& Z_tilde_ph,
                arma::mat& Z_ph,
                arma::cube& Z);

  void updateZTempered_PM(const double& beta_i,
                          const arma::field<arma::vec>& y_obs,
                          const arma::field<arma::mat>& y_star,
                          const arma::field<arma::mat>& B_obs,
                          const arma::field<arma::mat>& B_star,
                          const arma::cube& Phi,
                          const arma::mat& nu,
                          const arma::mat& chi,
                          const arma::vec& pi,
                          const double& sigma_sq,
                          const double& rho,
                          const int& iter,
                          const int& tot_mcmc_iters,
                          const double& alpha_3,
                          const double& sigma_Z,
                          arma::cube& Z_tilde,
                          arma::mat& Z_tilde_ph,
                          arma::mat& Z_ph,
                          arma::cube& Z);

#endif
