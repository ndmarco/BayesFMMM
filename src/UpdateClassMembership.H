#ifndef UPDATE_CLASS_MEMBERSHIP_H
#define UPDATE_CLASS_MEMBERSHIP_H

#include <RcppArmadillo.h>
#include <cmath>

double lpdf_z(const arma::vec& y_obs,
              const arma::mat& B_obs,
              const arma::cube& Phi,
              const arma::mat& nu,
              const arma::rowvec& chi,
              const arma::vec& pi,
              const arma::rowvec& Z,
              const double& sigma_sq);

double lpdf_zTempered(const double& beta_i,
                      const arma::vec& y_obs,
                      const arma::mat& B_obs,
                      const arma::cube& Phi,
                      const arma::mat& nu,
                      const arma::rowvec& chi,
                      const arma::vec& pi,
                      const arma::rowvec& Z,
                      const int& num,
                      const double& sigma_sq);

void updateZ(const arma::field<arma::vec>& y_obs,
             const arma::field<arma::mat>& B_obs,
             const arma::cube& Phi,
             const arma::mat& nu,
             const arma::mat& chi,
             const arma::vec& pi,
             const double& sigma_sq,
             const double& rho,
             const int& iter,
             const int& tot_mcmc_iters,
             arma::mat& Z_ph,
             arma::cube& Z);

void updateZ(const arma::field<arma::vec>& y_obs,
             const arma::field<arma::mat>& B_obs,
             const arma::cube& Phi,
             const arma::mat& nu,
             const arma::mat& chi,
             const arma::vec& pi,
             const double& sigma_sq,
             const double& rho,
             const int& n_known,
             const int& iter,
             const int& tot_mcmc_iters,
             arma::mat& Z_ph,
             arma::cube& Z);

void updateZTempered(const double& beta_i,
                     const arma::field<arma::vec>& y_obs,
                     const arma::field<arma::mat>& B_obs,
                     const arma::cube& Phi,
                     const arma::mat& nu,
                     const arma::mat& chi,
                     const arma::vec& pi,
                     const double& sigma_sq,
                     const double& rho,
                     const int& iter,
                     const int& tot_mcmc_iters,
                     arma::mat& Z_ph,
                     arma::cube& Z);

void updateZTempered(const double& beta_i,
                     const arma::field<arma::vec>& y_obs,
                     const arma::field<arma::mat>& B_obs,
                     const arma::cube& Phi,
                     const arma::mat& nu,
                     const arma::mat& chi,
                     const arma::vec& pi,
                     const double& sigma_sq,
                     const double& rho,
                     const int& n_known,
                     const int& iter,
                     const int& tot_mcmc_iters,
                     arma::mat& Z_ph,
                     arma::cube& Z);

#endif
