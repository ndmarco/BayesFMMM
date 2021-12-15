#ifndef BayesFPMM_UPDATE_PI_H
#define BayesFPMM_UPDATE_PI_H

#include <RcppArmadillo.h>
#include <cmath>
#include "Distributions.h"

namespace BayesFPMM{
// Updates pi
//
// @name UpdatePi
// @param alpha Double that is the hyperparameter
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int  containing total number of MCMC iterations
// @param Z Matrix that contains the current values of the binary matrix
// @param pi Matrix containg all samples of Pi
inline void updatePi(const double& alpha,
                     const arma::mat& Z,
                     const int& iter,
                     const int& tot_mcmc_iters,
                     arma::mat& pi){
  for(int l = 0; l < Z.n_cols; l++){
    pi(l, iter) = R::rbeta((alpha/ Z.n_cols) + arma::accu(Z.col(l)), Z.n_rows -
      arma::accu(Z.col(l)) + 1);
  }
  if(iter < (tot_mcmc_iters -1)){
    pi.col(iter + 1) = pi.col(iter);
  }
}

// Calculates the log pdf of the posterior distribution of pi
//
// @name lpdf_pi_PM
// @param c vector containing hyperparameters
// @param alpha_3 Double containing current value of alpha_3
// @param pi Vector containing pi parameters
// @param Z Matrix containing current Z parameters
// @return lpdf Double containing the log pdf
inline double lpdf_pi_PM(const arma::vec& c,
                         const double& alpha_3,
                         const arma::vec& pi,
                         const arma::mat& Z){
  double lpdf = 0;
  for(int k = 0; k < pi.n_elem; k++){
    lpdf = lpdf + ((c(k) - 1) * std::log(pi(k)));
    for(int i = 0; i <  Z.n_rows; i++){
      lpdf = lpdf + (((alpha_3 * pi(k)) - 1) * std::log(Z(i,k)));
    }
  }
  lpdf = lpdf -  (Z.n_rows * calc_lB(alpha_3 * pi));

  return lpdf;
}

// Calculates the probability that we propose a state given we are in another state
//
// @name pi_proposal_density
// @param pi Vector containing state we are proposing
// @param alpha Vector containing parameters used to propose the proposed state
inline double pi_proposal_density(const arma::vec& pi,
                                  const arma::vec& alpha)
{
  double density = 0;
  for(int i = 0; i < pi.n_elem; i++){
    density = density + (alpha(i) - 1) * std::log(pi(i));
  }

  density = density - calc_lB(alpha);

  return density;
}

// Updates pi for the partial membership model
//
// @name UpdatePi_PM
// @param alpha_3 Double containing the current value of alpha_3
// @param Z Matrix containing current value of Z parameters
// @param c Vector containing hyperparameters
// @param iter Int containing current MCMC iteration
// @param tot_mcmc_iters Int  containing total number of MCMC iterations
// @param a_pi_PM Double containing hyperparameter for sampling from pi
// @param pi_ph Vector containing placeholder for proposed update
// @param pi Matrix containing all values for pi values
inline void updatePi_PM(const double& alpha_3,
                        const arma::mat& Z,
                        const arma::vec& c,
                        const int& iter,
                        const int& tot_mcmc_iters,
                        const double& a_pi_PM,
                        arma::vec& pi_ph,
                        arma::mat& pi){

  pi_ph = rdirichlet(a_pi_PM * pi.col(iter));

  // calculate proposal log pdf
  double lpdf_new = lpdf_pi_PM(c, alpha_3, pi_ph, Z);

  // calculate current state log pdf
  double lpdf_old = lpdf_pi_PM(c, alpha_3, pi.col(iter), Z);

  double lpdf_propose_new = pi_proposal_density(pi_ph, a_pi_PM * pi.col(iter));
  double lpdf_propose_old = pi_proposal_density(pi.col(iter), a_pi_PM * pi_ph);

  double acceptance_prob = lpdf_new - lpdf_old + lpdf_propose_old - lpdf_propose_new;
  double rand_unif_var = R::runif(0,1);

  if(std::log(rand_unif_var) < acceptance_prob){
    // Accept new state and update parameters
    pi.col(iter) = pi_ph;
  }


  if((tot_mcmc_iters - 1) > iter){
    pi.col(iter+1) = pi.col(iter);
  }
}
}

#endif

