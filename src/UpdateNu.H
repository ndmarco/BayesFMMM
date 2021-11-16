#ifndef UPDATE_NU_H
#define UPDATE_NU_H

#include <RcppArmadillo.h>
#include <cmath>
#include <truncnorm.h>

void updateNu(const arma::field<arma::vec>& y_obs,
              const arma::field<arma::mat>& B_obs,
              const arma::vec& tau,
              const arma::cube& Phi,
              const arma::mat& Z,
              const arma::mat& chi,
              const double& sigma,
              const int& iter,
              const int& tot_mcmc_iters,
              const arma::mat& P,
              arma::vec& b_1,
              arma::mat& B_1,
              arma::cube& nu);

void updateNuTempered(const double& beta_i,
                      const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& B_obs,
                      const arma::vec& tau,
                      const arma::cube& Phi,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const double& sigma,
                      const int& iter,
                      const int& tot_mcmc_iters,
                      const arma::mat& P,
                      arma::vec& b_1,
                      arma::mat& B_1,
                      arma::cube& nu);

void updateNuMV(const arma::mat& y_obs,
                const arma::vec& tau,
                const arma::cube& Phi,
                const arma::mat& Z,
                const arma::mat& chi,
                const double& sigma,
                const int& iter,
                const int& tot_mcmc_iters,
                const arma::mat& P,
                arma::vec& b_1,
                arma::mat& B_1,
                arma::cube& nu);

void updateNuMVTempered(const double& beta_i,
                        const arma::mat& y_obs,
                        const arma::vec& tau,
                        const arma::cube& Phi,
                        const arma::mat& Z,
                        const arma::mat& chi,
                        const double& sigma,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        const arma::mat& P,
                        arma::vec& b_1,
                        arma::mat& B_1,
                        arma::cube& nu);
#endif

