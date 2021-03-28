#ifndef UPDATE_A_H
#define UPDATE_A_H

#include <RcppArmadillo.h>
#include <cmath>
#include <truncnorm.h>

double lpdf_a1(const double& alpha_1l,
               const double& beta_1l,
               const double& a,
               const double& delta);

double lpdf_a2(const double& alpha_2l,
               const double& beta_2l,
               const double& a,
               const arma::vec& delta);

void updateA(const double& alpha_1l,
             const double& beta_1l,
             const double& alpha_2l,
             const double& beta_2l,
             const arma::vec& delta,
             const double& var_epsilon1,
             const double& var_epsilon2,
             const int& iter,
             const int& tot_mcmc_iters,
             arma::mat& a);

#endif

