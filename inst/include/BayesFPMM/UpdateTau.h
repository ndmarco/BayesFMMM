#ifndef BayesFPMM_UPDATE_TAU_H
#define BayesFPMM_UPDATE_TAU_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFPMM{
// Updates the Tau parameters
//
// @name updateTau
// @param alpha Double containing hyperparameter
// @param beta Double containing hyperparameter
// @param nu Matrix contianing nu parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param P Matrix contiaing tridiagonal P matrix
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
// @param nu Matrix contianing nu parameters
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

}

#endif
