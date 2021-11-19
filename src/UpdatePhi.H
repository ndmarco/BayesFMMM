#ifndef UPDATE_PHI_H
#define UPDATE_PHI_H

#include <RcppArmadillo.h>
#include <cmath>

void updatePhi(const arma::field<arma::vec>& y_obs,
               const arma::field<arma::mat>& B_obs,
               const arma::mat& nu,
               const arma::cube& gamma,
               const arma::vec& tilde_tau,
               const arma::mat& Z,
               const arma::mat& chi,
               const double& sigma_sq,
               const int& iter,
               const int& tot_mcmc_iters,
               arma::vec& m_1,
               arma::mat& M_1,
               arma::field<arma::cube>& Phi);

void updatePhiTempered(const double& beta_i,
                       const arma::field<arma::vec>& y_obs,
                       const arma::field<arma::mat>& B_obs,
                       const arma::mat& nu,
                       const arma::cube& gamma,
                       const arma::vec& tilde_tau,
                       const arma::mat& Z,
                       const arma::mat& chi,
                       const double& sigma_sq,
                       const int& iter,
                       const int& tot_mcmc_iters,
                       arma::vec& m_1,
                       arma::mat& M_1,
                       arma::field<arma::cube>& Phi);

void updatePhiMV(const arma::mat& y_obs,
                 const arma::mat& nu,
                 const arma::cube& gamma,
                 const arma::vec& tilde_tau,
                 const arma::mat& Z,
                 const arma::mat& chi,
                 const double& sigma_sq,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 arma::vec& m_1,
                 arma::mat& M_1,
                 arma::field<arma::cube>& Phi);

void updatePhiTemperedMV(const double& beta_i,
                         const arma::mat& y_obs,
                         const arma::mat& nu,
                         const arma::cube& gamma,
                         const arma::vec& tilde_tau,
                         const arma::mat& Z,
                         const arma::mat& chi,
                         const double& sigma_sq,
                         const int& iter,
                         const int& tot_mcmc_iters,
                         arma::vec& m_1,
                         arma::mat& M_1,
                         arma::field<arma::cube>& Phi);
#endif
