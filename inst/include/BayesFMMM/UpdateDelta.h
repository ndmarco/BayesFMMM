#ifndef BayesFMMM_UPDATE_DELTA_H
#define BayesFMMM_UPDATE_DELTA_H

#include <RcppArmadillo.h>
#include <cmath>

namespace BayesFMMM{
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
        param1 = a(k,0) + (phi.n_cols  * phi.n_slices) / 2;
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
        param1 = a(k,1) + (phi.n_cols * (phi.n_slices - i)) / 2;
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


// Updates the delta parameters for individualized covariance matrix
//
// @name updateDelta
// @param phi Cube containing the current values of phi
// @param gamma Cube containing current values of gamma
// @param a_xi Cube containing current values of a_xi
// @param iter Int containing MCMC current iteration number
// @parma tot_mcmc_iters Int containing total number of MCMC iterations
// @param delta Cube containing values of delta
inline void updateDeltaXi(const arma::field<arma::cube>& xi,
                          const arma::field<arma::cube>& gamma_xi,
                          const arma::cube& a_xi,
                          const int& iter,
                          const int& tot_mcmc_iters,
                          arma::field<arma::cube>& delta){
  double param1 = 0;
  double param2 = 0;
  double tilde_tau = 0;
  for(int d = 0; d < delta.n_slices; d++){
    for(int k = 0; k < delta.n_rows; k++){
      for(int i = 0; i < delta.n_slices; i++){
        if(i == 0){
          param1 = a_xi(k,0,d) + (xi(iter,0).n_rows  * delta.n_rows) / 2;
          param2 = 1;
          for(int j = 0; j < xi(iter,0).n_rows; j++){
            param2 = param2 + (0.5 * gamma_xi(iter,k)(j, d, 0) * std::pow(xi(iter, k)(j, d, 0), 2));
            for(int m = 1; m < xi(iter, 0).n_slices; m++){
              tilde_tau = 1;
              for(int n = 1; n <= m; n++)
              {
                tilde_tau = tilde_tau * delta(iter, 0)(k, n, d);
              }
              param2 = param2 + (0.5 * gamma_xi(iter,k)(j, d, m) * tilde_tau * std::pow(xi(iter, k)(j, d, m), 2));
            }
          }
          delta(iter, 0)(k, i, d) = R::rgamma(param1, 1/param2);
        }else{
          param1 = a_xi(k,1,d) + (xi(iter,0).n_rows  * delta.n_rows) / 2;
          param2 = 1;
          for(int j = 0; j < xi(iter,0).n_rows; j++){
            for(int m = i; m < xi(iter, 0).n_slices; m++){
              tilde_tau = 1;
              for(int n = 0; n <= m; n++){
                if(n != i){
                  tilde_tau = tilde_tau * delta(iter, 0)(k, n, d);
                }
              }
              param2 = param2 + (0.5 * gamma_xi(iter,k)(j, d, m) * tilde_tau * std::pow(xi(iter, k)(j, d, m), 2));
            }
          }
          delta(iter, 0)(k, i, d) = R::rgamma(param1, 1/param2);
        }
      }
    }
  }

  if(iter < (tot_mcmc_iters - 1)){
    delta(iter + 1, 0) = delta(iter, 0);
  }
}
}

#endif
