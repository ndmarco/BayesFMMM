#ifndef BayesFMMM_UPDATE_A_H
#define BayesFMMM_UPDATE_A_H

#include <RcppArmadillo.h>
#include <cmath>
#include <truncnorm.h>
#include "Distributions.h"

namespace BayesFMMM{
// computes the log pdf of a_1j
//
// @name lpdf_a1
// @param alpha_1l Double containing hyperparameter for a
// @param beta_1l Double containing hyperparameter for a
// @param a Double containing value of a
// @param delta double contianing value of delta
inline double lpdf_a1(const double& alpha_1l,
                      const double& beta_1l,
                      const double& a,
                      const double& delta){
  double lpdf = -logGamma(a) + (a - 1) * log(delta) + (alpha_1l - 1) *
    log(a) - (a * beta_1l);
  return lpdf;
}

// computes the log pdf of a_2j
//
// @name lpdf_a2
// @param alpha_2l Double containing hyperparameter for a
// @param beta_2l Double containing hyperparameter for a
// @param a Double containing value of a
// @param delta double contianing value of delta
inline double lpdf_a2(const double& alpha_2l,
                      const double& beta_2l,
                      const double& a,
                      const arma::vec& delta){
  double x = delta.n_elem - 1;
  double lpdf = -x * logGamma(a) + (alpha_2l - 1) * log(a) -
    (a * beta_2l);
  for(int i = 1; i < delta.n_elem; i++){
    lpdf = lpdf + (a - 1) * log(delta(i));
  }
  return lpdf;
}

// updates the a parameters for individualized covariance matrix
//
// @name updateA
// @param alpha_1l Double containing hyperparameters for a
// @param beta_1l Double containing hyperparameters for a
// @param alpha_2l Double containing hyperparameters for a
// @param beta_2l Double containing hyperparameters for a
// @param delta Vector contianing value of delta
// @param var_epsilon1 Double containing hyperparameter epsilon1
// @param var_epsilon2 Double containing hyperparameter epsilon2
// @param iter Double containing MCMC iteration
// @param a Cube containing values of a
inline void updateA(const double& alpha_1l,
                    const double& beta_1l,
                    const double& alpha_2l,
                    const double& beta_2l,
                    const arma::mat& delta,
                    const double& var_epsilon1,
                    const double& var_epsilon2,
                    const int& iter,
                    const int& tot_mcmc_iters,
                    arma::cube& a){
  double a_lpdf = 0;
  double a_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;
  double new_a = 0;

  // calculate first lpdf
  for(int j = 0; j < a.n_rows; j++){
    for(int i = 0; i < a.n_cols; i++){
      if(i == 0){
        a_lpdf = lpdf_a1(alpha_1l, beta_1l, a(j, i, iter), delta(j,i));
        new_a = r_truncnorm(a(j, i, iter), var_epsilon1 / beta_1l, 0,
                            std::numeric_limits<double>::infinity());

        a_new_lpdf = lpdf_a1(alpha_1l, beta_1l, new_a, delta(j,i));

        acceptance_prob = (a_new_lpdf +
          d_truncnorm(a(j, i, iter), new_a, var_epsilon1 / beta_1l, 0,
                      std::numeric_limits<double>::infinity(), 1)) - a_lpdf -
                        d_truncnorm(new_a, a(j, i, iter),
                                    var_epsilon1 / beta_1l, 0,
                                    std::numeric_limits<double>::infinity(), 1);
        rand_unif_var = R::runif(0,1);

        if(log(rand_unif_var) < acceptance_prob){
          // Accept new state and update parameters
          a(j, i, iter) = new_a;
        }
      }else{
        a_lpdf = lpdf_a2(alpha_2l, beta_2l, a(j, i, iter), delta.row(j).t());
        new_a = r_truncnorm(a(j, i, iter), var_epsilon2 / beta_2l, 0,
                            std::numeric_limits<double>::infinity());

        a_new_lpdf = lpdf_a2(alpha_2l, beta_2l, new_a, delta.row(j).t());

        acceptance_prob = (a_new_lpdf +
          d_truncnorm(a(j, i, iter), new_a, var_epsilon2 / beta_2l, 0,
                      std::numeric_limits<double>::infinity(), 1)) - a_lpdf -
                        d_truncnorm(new_a, a(j, i, iter),
                                    var_epsilon2 / beta_2l, 0,
                                    std::numeric_limits<double>::infinity(), 1);
        rand_unif_var = R::runif(0,1);

        if(log(rand_unif_var) < acceptance_prob){
          // Accept new state and update parameters
          a(j, i, iter) = new_a;
        }
      }
    }
  }

  // update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    a.slice(iter + 1) = a.slice(iter);
  }
}

// updates the a parameters for individualized covariance matrix
//
// @name updateA
// @param alpha_1l Double containing hyperparameters for a
// @param beta_1l Double containing hyperparameters for a
// @param alpha_2l Double containing hyperparameters for a
// @param beta_2l Double containing hyperparameters for a
// @param delta Vector contianing value of delta
// @param var_epsilon1 Double containing hyperparameter epsilon1
// @param var_epsilon2 Double containing hyperparameter epsilon2
// @param iter Double containing MCMC iteration
// @param a Cube containing values of a
inline void updateAXi(const double& alpha_1l,
                      const double& beta_1l,
                      const double& alpha_2l,
                      const double& beta_2l,
                      const arma::cube& delta,
                      const double& var_epsilon1,
                      const double& var_epsilon2,
                      const int& iter,
                      const int& tot_mcmc_iters,
                      arma::field<arma::cube>& a){
  double a_lpdf = 0;
  double a_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;
  double new_a = 0;

  // calculate first lpdf
  for(int j = 0; j < a(iter,0).n_rows; j++){
    for(int i = 0; i < a(iter,0).n_cols; i++){
      for(int d = 0; d < a(iter,0).n_slices; d++){
        if(i == 0){
          a_lpdf = lpdf_a1(alpha_1l, beta_1l, a(iter,0)(j, i, d), delta(j,i,d));
          new_a = r_truncnorm(a(iter,0)(j, i, d), var_epsilon1 / beta_1l, 0,
                              std::numeric_limits<double>::infinity());

          a_new_lpdf = lpdf_a1(alpha_1l, beta_1l, new_a, delta(j,i,d));

          acceptance_prob = (a_new_lpdf +
            d_truncnorm(a(iter,0)(j, i, d), new_a, var_epsilon1 / beta_1l, 0,
                        std::numeric_limits<double>::infinity(), 1)) - a_lpdf -
                          d_truncnorm(new_a, a(iter,0)(j, i, d),
                                      var_epsilon1 / beta_1l, 0,
                                      std::numeric_limits<double>::infinity(), 1);
          rand_unif_var = R::runif(0,1);

          if(log(rand_unif_var) < acceptance_prob){
            // Accept new state and update parameters
            a(iter,0)(j, i, d) = new_a;
          }
        }else{

          a_lpdf = lpdf_a2(alpha_2l, beta_2l, a(iter,0)(j, i, d), delta.slice(d).row(j).t());
          new_a = r_truncnorm(a(iter,0)(j, i, d), var_epsilon2 / beta_2l, 0,
                              std::numeric_limits<double>::infinity());

          a_new_lpdf = lpdf_a2(alpha_2l, beta_2l, new_a, delta.slice(d).row(j).t());

          acceptance_prob = (a_new_lpdf +
            d_truncnorm(a(iter,0)(j, i, d), new_a, var_epsilon2 / beta_2l, 0,
                        std::numeric_limits<double>::infinity(), 1)) - a_lpdf -
                          d_truncnorm(new_a, a(iter,0)(j, i, d),
                                      var_epsilon2 / beta_2l, 0,
                                      std::numeric_limits<double>::infinity(), 1);
          rand_unif_var = R::runif(0,1);

          if(log(rand_unif_var) < acceptance_prob){
            // Accept new state and update parameters
            a(iter,0)(j, i, d) = new_a;
          }
        }
      }
    }
  }

  // update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    a(iter + 1, 0) = a(iter, 0);
  }
}

}
#endif

