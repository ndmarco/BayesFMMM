#ifndef BayesFPMM_UPDATE_SIGMA_H
#define BayesFPMM_UPDATE_SIGMA_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFPMM{
// Updates the Sigma parameters
//
// @name updateSigma
// @param y_obs Field of vectors containing observed time points
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param alpha_0 Double containing hyperparameter
// @param beta_0 Double containing hyperparameter
// @param nu Matrix containing current nu parameters
// @param Phi Cube containing current Phi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param sigma Vector containing sigma for all mcmc iterations
inline void updateSigma(const arma::field<arma::vec>& y_obs,
                        const arma::field<arma::mat>& B_obs,
                        const double alpha_0,
                        const double beta_0,
                        const arma::mat& nu,
                        const arma::cube& Phi,
                        const arma::mat& Z,
                        const arma::mat& chi,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        arma::vec& sigma){
  double a = 0;
  double b = 0;
  double b_1 = 0;
  for(int i = 0; i < Z.n_rows; i++){
    for(int l = 0; l < y_obs(i,0).n_elem; l++){
      b = y_obs(i,0)(l);
      for(int k = 0; k < Z.n_cols; k++){
        if(Z(i,k) != 0){
          b = b - Z(i,k) * arma::dot(nu.row(k), B_obs(i,0).row(l));
          for(int n = 0; n < Phi.n_slices; n++){
            b = b - Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k), B_obs(i,0).row(l));
          }
        }
      }
      b_1 = b_1 + 0.5 * (b * b);
    }
    a = a + (y_obs(i,0).n_elem / 2);
  }
  b_1 = b_1 + beta_0;
  a = a + alpha_0;
  sigma(iter) = 1 / R::rgamma(a, 1/b_1);

  if(iter < (tot_mcmc_iters - 1)){
    sigma(iter + 1) = sigma(iter);
  }
}

// Updates the Sigma parameters using Tempered Transitions
//
// @name updateSigmaTempered
// @param beta_i Double containing current temperature
// @param y_obs Field of vectors containing observed time points
// @param B_obs Field of matrices containing basis functions evaluated at observed time points
// @param alpha_0 Double containing hyperparameter
// @param beta_0 Double containing hyperparameter
// @param nu Matrix containing current nu parameters
// @param Phi Cube containing current Phi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param sigma Vector containing sigma for all mcmc iterations
inline void updateSigmaTempered(const double& beta_i,
                                const arma::field<arma::vec>& y_obs,
                                const arma::field<arma::mat>& B_obs,
                                const double alpha_0,
                                const double beta_0,
                                const arma::mat& nu,
                                const arma::cube& Phi,
                                const arma::mat& Z,
                                const arma::mat& chi,
                                const int& iter,
                                const int& tot_mcmc_iters,
                                arma::vec& sigma){
  double a = 0;
  double b = 0;
  double b_1 = 0;
  for(int i = 0; i < Z.n_rows; i++){
    for(int l = 0; l < y_obs(i,0).n_elem; l++){
      b = y_obs(i,0)(l);
      for(int k = 0; k < Z.n_cols; k++){
        if(Z(i,k) != 0){
          b = b - Z(i,k) * arma::dot(nu.row(k), B_obs(i,0).row(l));
          for(int n = 0; n < Phi.n_slices; n++){
            b = b - Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k), B_obs(i,0).row(l));
          }
        }
      }
      b_1 = b_1 + (beta_i / 2) * (b * b);
    }
    a = a + ((beta_i * y_obs(i,0).n_elem) / 2);
  }
  b_1 = b_1 + beta_0;
  a = a + alpha_0;
  sigma(iter) = 1 / R::rgamma(a, 1/b_1);

  if(iter < (tot_mcmc_iters - 1)){
    sigma(iter + 1) = sigma(iter);
  }
}

// Updates the Sigma parameters for the multivariate model
//
// @name updateSigmaMV
// @param y_obs matrix containing observed vectors
// @param alpha_0 Double containing hyperparameter
// @param beta_0 Double containing hyperparameter
// @param nu Matrix containing current nu parameters
// @param Phi Cube containing current Phi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param sigma Vector containing sigma for all mcmc iterations
inline void updateSigmaMV(const arma::mat& y_obs,
                          const double alpha_0,
                          const double beta_0,
                          const arma::mat& nu,
                          const arma::cube& Phi,
                          const arma::mat& Z,
                          const arma::mat& chi,
                          const int& iter,
                          const int& tot_mcmc_iters,
                          arma::vec& sigma){
  double b_1 = 0;
  arma::vec mean = arma::zeros(nu.n_cols);
  for(int i = 0; i < Z.n_rows; i++){
    mean = arma::zeros(nu.n_cols);
    for(int k = 0; k < Z.n_elem; k++){
      mean = mean + Z(i,k) * nu.row(k).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(i,k) * chi(i,m) * Phi.slice(m).row(k).t();
      }
    }
    b_1 = b_1 + 0.5 * arma::dot(y_obs.row(i).t() - mean, y_obs.row(i).t() - mean);
  }
  b_1 = b_1 + beta_0;
  double a = (y_obs.n_elem / 2) + alpha_0;
  sigma(iter) = 1 / R::rgamma(a, 1/b_1);

  if(iter < (tot_mcmc_iters - 1)){
    sigma(iter + 1) = sigma(iter);
  }
}

// Updates the Sigma parameters using Tempered Transitions
//
// @name updateSigma
// @param beta_i Double containing current temperature
// @param y_obs matrix containing observed vectors
// @param alpha_0 Double containing hyperparameter
// @param beta_0 Double containing hyperparameter
// @param nu Matrix containing current nu parameters
// @param Phi Cube containing current Phi parameters
// @param Z Matrix containing current Z parameters
// @param chi Matrix containing current chi parameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param sigma Vector containing sigma for all mcmc iterations
inline void updateSigmaTemperedMV(const double& beta_i,
                                  const arma::mat& y_obs,
                                  const double alpha_0,
                                  const double beta_0,
                                  const arma::mat& nu,
                                  const arma::cube& Phi,
                                  const arma::mat& Z,
                                  const arma::mat& chi,
                                  const int& iter,
                                  const int& tot_mcmc_iters,
                                  arma::vec& sigma){
  double b_1 = 0;
  arma::vec mean = arma::zeros(nu.n_cols);
  for(int i = 0; i < Z.n_rows; i++){
    mean = arma::zeros(nu.n_cols);
    for(int k = 0; k < Z.n_elem; k++){
      mean = mean + Z(i,k) * nu.row(k).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(i,k) * chi(i,m) * Phi.slice(m).row(k).t();
      }
    }
    b_1 = b_1 + (beta_i / 2) *
      arma::dot(y_obs.row(i).t() - mean, y_obs.row(i).t() - mean);
  }
  b_1 = b_1 + beta_0;
  double a = ((beta_i * y_obs.n_elem) / 2) + alpha_0;
  sigma(iter) = 1 / R::rgamma(a, 1/b_1);

  if(iter < (tot_mcmc_iters - 1)){
    sigma(iter + 1) = sigma(iter);
  }
}

}
#endif
