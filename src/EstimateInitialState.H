#ifndef ESTIMATE_INITIAL_STATE_H
#define ESTIMATE_INITIAL_STATE_H

#include <RcppArmadillo.h>
#include<cmath>

arma::mat BasisExpansion(const arma::field<arma::vec>& y_obs,
                         const arma::field<arma::mat>& B_obs,
                         const int& n_funct,
                         const int& K,
                         const int& P);

double SigmaInitialState(const arma::field<arma::vec>& y_obs,
                         const arma::field<arma::mat>& B_obs,
                         const arma::mat& theta,
                         const int& n_funct);

arma::mat NuInitialState(const arma::field<arma::mat>& B_obs,
                         const arma::mat& Z_known,
                         const arma::mat& theta,
                         const int& n_funct);

arma::mat ZInitialState(const arma::field<arma::mat>& B_obs,
                        const arma::mat& theta,
                        const double& alpha,
                        const int& max_iter,
                        const int& K,
                        const int& n_funct,
                        const double& convergence);

Rcpp::List PhiChiInitialState(const arma::mat& known_Z,
                              const arma::field<arma::vec>& y_obs,
                              const arma::field<arma::vec>& t_obs,
                              const double& n_funct,
                              const int& K,
                              const int& P,
                              const int& M,
                              const int& tot_mcmc_iters,
                              const int& r_stored_iters,
                              const arma::field<arma::vec>& t_star,
                              const double& nu_1,
                              const double& rho,
                              const double& alpha_3,
                              const double& alpha1l,
                              const double& alpha2l,
                              const double& beta1l,
                              const double& beta2l,
                              const double& var_epsilon1,
                              const double& var_epsilon2,
                              const double& alpha,
                              const double& beta,
                              const double& alpha_0,
                              const double& beta_0,
                              const arma::mat& nu,
                              const double& sigma);


#endif
