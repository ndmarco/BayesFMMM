#ifndef BayesFMMM_UPDATE_TAU_H
#define BayesFMMM_UPDATE_TAU_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFMMM{
// Updates the Tau parameters
//
// @name updateTau
// @param alpha Double containing hyperparameter
// @param beta Double containing hyperparameter
// @param nu Matrix containing nu parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param P Matrix containing tridiagonal P matrix
// @param tau Matrix containing tau for all mcmc iterations
inline void updateTau(const double& alpha,
                      const double& beta,
                      const arma::mat& nu,
                      const int& iter,
                      const int& tot_mcmc_iters,
                      const arma::mat& P,
                      arma::mat& tau){
  double a = 0;
  double b = 0;

  for(int i = 0; i < tau.n_cols; i++){
    a = alpha + (nu.n_cols / 2);
    b = beta + (0.5 * arma::dot(nu.row(i), P * nu.row(i).t()));
    tau(iter, i) =  R::rgamma(a, 1/b);
  }
  if(iter < (tot_mcmc_iters - 1)){
    tau.row(iter + 1) = tau.row(iter);
  }
}

// Updates the Tau parameters for the multivariate model
//
// @name updateTauMV
// @param alpha Double containing hyperparameter
// @param beta Double containing hyperparameter
// @param nu Matrix containing nu parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param tau Matrix containing tau for all mcmc iterations
inline void updateTauMV(const double& alpha,
                        const double& beta,
                        const arma::mat& nu,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        arma::mat& tau){
  double a = 0;
  double b = 0;
  for(int i = 0; i < tau.n_cols; i++){
    a = alpha + (nu.n_cols / 2);
    b = beta + (0.5 * arma::dot(nu.row(i), nu.row(i).t()));
    tau(iter, i) =  1 / R::rgamma(a, 1/b);
  }
  if(iter < (tot_mcmc_iters - 1)){
    tau.row(iter + 1) = tau.row(iter);
  }
}

// Updates the Tau_eta parameters for covariate adjusted model
//
// @name updateTauEta
// @param alpha Double containing hyperparameter
// @param beta Double containing hyperparameter
// @param eta Cube containing eta parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param P Matrix containing tridiagonal P matrix
// @param tau_eta Cube containing tau_eta for all mcmc iterations
inline void updateTauEta(const double& alpha,
                         const double& beta,
                         const arma::cube& eta,
                         const int& iter,
                         const int& tot_mcmc_iters,
                         const arma::mat& P,
                         arma::cube& tau_eta){
  double a = 0;
  double b = 0;

  for(int j = 0; j < tau_eta.n_rows; j++){
    for(int i = 0; i < tau_eta.n_cols; i++){
      a = alpha + (eta.n_rows / 2);
      b = beta + (0.5 * arma::dot(eta.slice(j).col(i), P * eta.slice(j).col(i)));
      tau_eta(j, i, iter) =  R::rgamma(a, 1/b);
    }
  }
  if(iter < (tot_mcmc_iters - 1)){
    tau_eta.slice(iter + 1) = tau_eta.slice(iter);
  }
}

// Updates the Tau parameters for the multivariate covariate adjusted model
//
// @name updateTauEtaMV
// @param alpha Double containing hyperparameter
// @param beta Double containing hyperparameter
// @param eta Cube containing eta parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param tau_eta Cube containing tau for all mcmc iterations
inline void updateTauEtaMV(const double& alpha,
                        const double& beta,
                        const arma::cube& eta,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        arma::cube& tau_eta){
  double a = 0;
  double b = 0;
  for(int j = 0; j < tau_eta.n_rows; j++){
    for(int i = 0; i < tau_eta.n_cols; i++){
      a = alpha + (eta.n_rows / 2);
      b = beta + (0.5 * arma::dot(eta.slice(j).col(i), eta.slice(j).col(i)));
      tau_eta(j, i, iter) =  1 / R::rgamma(a, 1/b);
    }
  }
  if(iter < (tot_mcmc_iters - 1)){
    tau_eta.slice(iter + 1) = tau_eta.slice(iter);
  }
}

}

#endif
