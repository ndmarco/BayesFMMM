#ifndef BayesFMMM_UPDATE_GAMMA_H
#define BayesFMMM_UPDATE_GAMMA_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFMMM{
// Updates the gamma parameters
//
// @name updateGamma
// @param nu_gamma double containing hyperparameter
// @param iter int containing MCMC iteration
// @param delta Matrix containing current values of delta
// @param phi Cube containing current values of phi
// @param Z matrix containing current values of class inclusion
// @param gamma Field of cubes contianing MCMC samples for gamma
inline void updateGamma(const double& nu_gamma,
                        const arma::mat& delta,
                        const arma::cube& phi,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        arma::field<arma::cube>& gamma){
  double placeholder = 1;
  for(int i = 0; i < phi.n_rows; i++){
    for(int l = 0; l < phi.n_cols; l++){
      placeholder = 1;
      for(int j = 0; j < phi.n_slices; j++){
        placeholder = placeholder * delta(i, j);
        gamma(iter,0)(i,l,j) = R::rgamma((nu_gamma + 1)/2, 2/(nu_gamma + placeholder *
          (phi(i,l,j) * phi(i,l,j))));
      }
    }
  }
  if(iter < (tot_mcmc_iters) - 1){
    gamma(iter + 1, 0) = gamma(iter, 0);
  }
}

// Updates the gamma_xi parameters for the covariate adjusted model
//
// @name updateGammaXi
// @param nu_gamma double containing hyperparameter
// @param iter int containing MCMC iteration
// @param delta Matrix containing current values of delta
// @param phi Cube containing current values of phi
// @param Z matrix containing current values of class inclusion
// @param gamma Field of cubes contianing MCMC samples for gamma
inline void updateGammaXi(const double& nu_gamma,
                          const arma::cube& delta_xi,
                          const arma::field<arma::cube>& xi,
                          const int& iter,
                          const int& tot_mcmc_iters,
                          arma::field<arma::cube>& gamma_xi){
  double placeholder = 1;
  for(int k = 0; k < delta_xi.n_rows; k++){
    for(int i = 0; i < gamma_xi(iter,0).n_cols; i++){
      for(int l = 0; l < gamma_xi(iter,0).n_rows; l++){
        placeholder = 1;
        for(int j = 0; j < gamma_xi(iter,0).n_slices; j++){
          placeholder = placeholder * delta_xi(k, j, i);
          gamma_xi(iter,k)(l,i,j) = R::rgamma((nu_gamma + 1)/2, 2/(nu_gamma + placeholder *
            (xi(iter,k)(l,i,j) * xi(iter,k)(l,i,j))));
        }
      }
    }
  }

  if(iter < (tot_mcmc_iters) - 1){
    for(int k = 0; k < delta_xi.n_rows; k++){
      gamma_xi(iter + 1, k) = gamma_xi(iter, k);
    }
  }
}
}
#endif
