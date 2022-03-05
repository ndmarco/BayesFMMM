#ifndef BayesFPMM_UPDATE_DELTA_H
#define BayesFPMM_UPDATE_DELTA_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFPMM{
// Updates the delta parameters for individualized covariance matrix
//
// @name updateDelta
// @param phi Cube containing the current values of phi
// @param gamma Cube containing current values of gamma
// @param a Cube containing current values of a
// @param iter Int containing MCMC current iteration number
// @parma tot_mcmc_iters Int containing total number of MCMC iterations
// @param delta Cube containing values of delta
inline void updateDelta(const arma::cube& phi,
                        const arma::cube& gamma,
                        const arma::mat& a,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        arma::cube& delta){
  double param1 = 0;
  double param2 = 0;
  double tilde_tau = 0;
  for(int k = 0; k < phi.n_rows; k++){
    for(int i = 0; i < phi.n_slices; i++){
      if(i == 0){
        param1 = a(0) + (phi.n_cols  * phi.n_slices) / 2;
        param2 = 1;
        for(int j = 0; j < phi.n_cols; j++){
          param2 = param2 + (0.5 * gamma(k, j, 0) * std::pow(phi(k, j, 0), 2));
          for(int m = 1; m < phi.n_slices; m++){
            tilde_tau = 1;
            for(int n = 1; n <= m; n++)
            {
              tilde_tau = tilde_tau * delta(k, n, iter);
            }
            param2 = param2 + (0.5 * gamma(k, j, m) * tilde_tau * std::pow(phi(k, j, m), 2));
          }
        }
        delta(k, i, iter) = R::rgamma(param1, 1/param2);
      }else{
        param1 = a(1) + (phi.n_cols * (phi.n_slices - i)) / 2;
        param2 = 1;
        for(int j = 0; j < phi.n_cols; j++){
          for(int m = i; m < phi.n_slices; m++){
            tilde_tau = 1;
            for(int n = 0; n <= m; n++){
              if(n != i){
                tilde_tau = tilde_tau * delta(k, n, iter);
              }
            }
            param2 = param2 + (0.5 * gamma(k, j, m) * tilde_tau * std::pow(phi(k, j, m), 2));
          }
        }
        delta(k, i, iter) = R::rgamma(param1, 1/param2);
      }
    }
  }
  if(iter < (tot_mcmc_iters - 1)){
    delta.slice(iter + 1) = delta.slice(iter);
  }
}
}

#endif
