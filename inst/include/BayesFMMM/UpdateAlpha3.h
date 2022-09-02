#ifndef BayesFMMM_UPDATE_ALPHA_3_H
#define BayesFMMM_UPDATE_ALPHA_3_H

#include <RcppArmadillo.h>
#include <cmath>
#include <truncnorm.h>
#include "Distributions.h"

namespace BayesFMMM{
inline double lpdf_alpha3(const arma::vec& pi,
                          const double& b,
                          const arma::mat& Z,
                          double& alpha_3,
                          double& alpha_3_ph,
                          const double& sigma_alpha_3){
  double lpdf = (-b) * alpha_3;
  for(int k = 0; k < Z.n_cols; k++){
    for(int i = 0; i < Z.n_rows; i++){
      lpdf = lpdf + (((alpha_3 * pi(k)) - 1) * std::log(Z(i,k)));
    }
  }
  lpdf = lpdf - (Z.n_rows *calc_lB(alpha_3 * pi));
  lpdf = lpdf + d_truncnorm(alpha_3_ph, alpha_3_ph, sigma_alpha_3, 0,
                            std::numeric_limits<double>::infinity(), 1);
  return lpdf;
}


// Updates the Alpha3 parameter
//
// @name updateAlpha3
// @param pi Vector containing current values of pi
// @param b Double containing hyperparameter b
// @param Z Matrix containing current values of Z
// @param alpha_3 vector containing all alpha_3
inline void updateAlpha3(const arma::vec& pi,
                         const double& b,
                         const arma::mat& Z,
                         const int& iter,
                         const int& tot_mcmc_iters,
                         const double& sigma_alpha_3,
                         arma::vec& alpha_3){

  // propose new value
  double alpha_3_ph = r_truncnorm(alpha_3(iter), sigma_alpha_3, 0,
                                  std::numeric_limits<double>::infinity());

  double lpdf_old = lpdf_alpha3(pi, b, Z, alpha_3(iter), alpha_3_ph, sigma_alpha_3);

  double lpdf_new = lpdf_alpha3(pi, b, Z, alpha_3_ph, alpha_3(iter), sigma_alpha_3);

  double acceptance_prob = lpdf_new - lpdf_old;
  double rand_unif_var = R::runif(0,1);

  if(std::log(rand_unif_var) < acceptance_prob){
    // Accept new state and update parameters
    alpha_3(iter) = alpha_3_ph;
  }

  if((tot_mcmc_iters - 1) > iter){
    alpha_3(iter+1) = alpha_3(iter);
  }
}
}

#endif
