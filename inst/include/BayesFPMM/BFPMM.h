#ifndef BayesFPMM_BFPMM_H
#define BayesFPMM_BFPMM_H

#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "UpdateClassMembership.h"
#include "UpdatePartialMembership.h"
#include "UpdatePi.h"
#include "UpdatePhi.h"
#include "UpdateDelta.h"
#include "UpdateA.h"
#include "UpdateGamma.h"
#include "UpdateNu.h"
#include "UpdateTau.h"
#include "UpdateSigma.h"
#include "UpdateChi.h"
#include "CalculateLikelihood.h"
#include "CalculateTTAcceptance.h"
#include "UpdateAlpha3.h"
#include "BSplines.h"
#include "Distributions.h"

namespace BayesFPMM {

// Conducts un-tempered MCMC to estimate the posterior distribution in an unsupervised setting. MCMC samples will be stored in batches to a specified path.
//
// @name BFPMM
// @param y_obs Field (list) of vectors containing the observed values
// @param t_obs Field (list) of vectors containing time points of observed values
// @param n_funct Int containing number of functions observed
// @param thinning_num Int that saves every (thinning_num) sample
// @param P Int that indicates the number of b-spline basis functions
// @param M int that indicates the number of slices used in Phi parameter
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int constaining number of iterations performed for each batch
// @param c Vector containing hyperparmeters for pi
// @param b double containing hyperparameter for alpha_3
// @param a_12 Vector containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param var_pi Double containing variance parameter of the random walk MH for pi parameter
// @param var_Z Double containing variance parameter of the random walk MH for Z parameter
// @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param directory String containing path to store batches of MCMC samples
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM(const arma::field<arma::vec>& y_obs,
                        const arma::field<arma::vec>& t_obs,
                        const int& n_funct,
                        const int& thinning_num,
                        const int& K,
                        const int& P,
                        const int& M,
                        const int& tot_mcmc_iters,
                        const int& r_stored_iters,
                        const arma::vec& c,
                        const double& b,
                        const double& nu_1,
                        const double& alpha1l,
                        const double& alpha2l,
                        const double& beta1l,
                        const double& beta2l,
                        const double& a_Z_PM,
                        const double& a_pi_PM,
                        const double& var_alpha3,
                        const double& var_epsilon1,
                        const double& var_epsilon2,
                        const double& alpha,
                        const double& beta,
                        const double& alpha_0,
                        const double& beta_0,
                        const std::string directory){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);

  for(int i = 0; i < n_funct; i++){
    splines2::BSpline bspline;
    // Create Bspline object with 8 degrees of freedom
    // 8 - 3 - 1 internal nodes
    bspline = splines2::BSpline(t_obs(i,0), P);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat {bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  arma::mat P_mat(P, P, arma::fill::zeros);
  P_mat.zeros();
  for(int j = 0; j < P_mat.n_rows; j++){
    P_mat(0,0) = 1;
    if(j > 0){
      P_mat(j,j) = 2;
      P_mat(j-1,j) = -1;
      P_mat(j,j-1) = -1;
    }
    P_mat(P_mat.n_rows - 1, P_mat.n_rows - 1) = 1;
  }

  arma::cube nu(K, P, r_stored_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(r_stored_iters);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_funct; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  for(int i = 0; i < tot_mcmc_iters; i++){
    updateZ_PM(y_obs, B_obs, Phi((i % r_stored_iters),0),
               nu.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
               pi.col((i % r_stored_iters)), sigma((i % r_stored_iters)),
               (i % r_stored_iters), r_stored_iters, alpha_3(i % r_stored_iters),
               a_Z_PM, Z_ph, Z);
    updatePi_PM(alpha_3(i % r_stored_iters) ,Z.slice(i% r_stored_iters), c,
                (i % r_stored_iters), r_stored_iters, a_pi_PM, pi_ph, pi);

    updateAlpha3(pi.col(i % r_stored_iters), b, Z.slice(i % r_stored_iters),
                 (i % r_stored_iters), r_stored_iters, var_alpha3, alpha_3);

    tilde_tau(0) = delta(0, (i % r_stored_iters));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i % r_stored_iters));
    }

    updatePhi(y_obs, B_obs, nu.slice((i % r_stored_iters)),
              gamma((i % r_stored_iters),0), tilde_tau,
              Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
              sigma((i % r_stored_iters)), (i % r_stored_iters),
              r_stored_iters, m_1, M_1, Phi);

    updateDelta(Phi((i % r_stored_iters),0), gamma((i % r_stored_iters),0),
                A.row(i % r_stored_iters).t(), (i % r_stored_iters),
                r_stored_iters, delta);

    updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i % r_stored_iters)),
            var_epsilon1, var_epsilon2, (i % r_stored_iters), r_stored_iters, A);

    updateGamma(nu_1, delta.col((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                (i % r_stored_iters), r_stored_iters, gamma);

    updateNu(y_obs, B_obs, tau.row((i % r_stored_iters)).t(),
             Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)),
             chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)),
             (i % r_stored_iters), r_stored_iters, P_mat, b_1, B_1, nu);

    updateTau(alpha, beta, nu.slice((i % r_stored_iters)), (i % r_stored_iters),
              r_stored_iters, P_mat, tau);

    updateSigma(y_obs, B_obs, alpha_0, beta_0,
                nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                (i % r_stored_iters), r_stored_iters, sigma);

    updateChi(y_obs, B_obs, Phi((i % r_stored_iters),0),
              nu.slice((i % r_stored_iters)), Z.slice((i % r_stored_iters)),
              sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters,
              chi);

    // Calculate log likelihood
    loglik((i % r_stored_iters)) =  calcLikelihood(y_obs, B_obs, nu.slice((i % r_stored_iters)),
           Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)));
    if(((i+1) % 20) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i % r_stored_iters)-19, (i % r_stored_iters))) << "\n";
      Rcpp::checkUserInterrupt();
    }
    if(((i+1) % r_stored_iters) == 0 && i > 1){
      // Save parameters
      arma::cube nu1(K, P, r_stored_iters/thinning_num, arma::fill::randn);
      arma::cube chi1(n_funct, M, r_stored_iters/thinning_num, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec alpha_31(r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters/thinning_num, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters/thinning_num, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_funct, K, r_stored_iters/thinning_num,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters/thinning_num, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters/thinning_num,1);
      arma::field<arma::cube> Phi1(r_stored_iters/thinning_num, 1);
      arma::mat tau1(r_stored_iters/thinning_num, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);

      for(int p=1; p < r_stored_iters / thinning_num; p++){
        nu1.slice(p) = nu.slice(thinning_num*p - 1);
        chi1.slice(p) = chi.slice(thinning_num*p - 1);
        pi1.col(p) = pi.col(thinning_num*p - 1);
        alpha_31(p) = alpha_3(thinning_num*p - 1);
        sigma1(p) = sigma(thinning_num*p - 1);
        A1.row(p) = A.row(thinning_num*p - 1);
        Z1.slice(p) = Z.slice(thinning_num*p - 1);
        delta1.col(p) = delta.col(thinning_num*p - 1);
        gamma1(p,0) = gamma(thinning_num*p - 1,0);
        Phi1(p,0) = Phi(thinning_num*p  - 1,0);
        tau1.row(p) = tau.row(thinning_num*p - 1);
      }

      nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
      chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
      pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
      alpha_31.save(directory + "alpha_3" + std::to_string(q) + ".txt", arma::arma_ascii);
      A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
      delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
      sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
      tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
      gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
      Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
      Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

      //reset all parameters
      nu.slice(0) = nu.slice(i % r_stored_iters);
      chi.slice(0) = chi.slice(i % r_stored_iters);
      pi.col(0) = pi.col(i % r_stored_iters);
      alpha_3(0) = alpha_3(i % r_stored_iters);
      A.row(0) = A.row(i % r_stored_iters);
      delta.col(0) = delta.col(i % r_stored_iters);
      sigma(0) = sigma(i % r_stored_iters);
      tau.row(0) = tau.row(i % r_stored_iters);
      gamma(0,0) = gamma(i % r_stored_iters, 0);
      Phi(0,0) = Phi(i % r_stored_iters, 0);
      Z.slice(0) = Z.slice(i % r_stored_iters);

      q = q + 1;
    }
  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
                                         Rcpp::Named("chi", chi),
                                         Rcpp::Named("pi", pi),
                                         Rcpp::Named("alpha_3", alpha_3),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("gamma", gamma),
                                         Rcpp::Named("Phi", Phi),
                                         Rcpp::Named("Z", Z),
                                         Rcpp::Named("loglik", loglik));
  return params;
}

// Conducts tempered MCMC to estimate the posterior distribution in an unsupervised setting. MCMC samples will be stored in batches to a specified path.
//
// @name BFPMM_Templadder
// @param y_obs Field (list) of vectors containing the observed values
// @param t_obs Field (list) of vectors containing time points of observed values
// @param n_funct Int containing number of functions observed
// @param P Int that indicates the number of b-spline basis functions
// @param M int that indicates the number of slices used in Phi parameter
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int constaining number of iterations performed for each batch
// @param rho Double containing hyperparmater for sampling from Z
// @param alpha_3 Double hyperparameter for sampling from pi
// @param a_12 Vec containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param a_Z_PM Double containing hyperparameter used to sample from the posterior of Z
// @param a_pi_PM Double containing hyperparameter used to sample from the posterior of pi
// @param var_alpha3 Doubel containing hyperparameter for sampling from alpha_3
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param directory String containing path to store batches of MCMC samples
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_Templadder(const arma::field<arma::vec>& y_obs,
                                   const arma::field<arma::vec>& t_obs,
                                   const int& n_funct,
                                   const int& K,
                                   const int& P,
                                   const int& M,
                                   const int& tot_mcmc_iters,
                                   const int& r_stored_iters,
                                   const arma::vec& c,
                                   const double& b,
                                   const double& nu_1,
                                   const double& alpha1l,
                                   const double& alpha2l,
                                   const double& beta1l,
                                   const double& beta2l,
                                   const double& a_Z_PM,
                                   const double& a_pi_PM,
                                   const double& var_alpha3,
                                   const double& var_epsilon1,
                                   const double& var_epsilon2,
                                   const double& alpha,
                                   const double& beta,
                                   const double& alpha_0,
                                   const double& beta_0,
                                   const double& beta_N_t,
                                   const int& N_t){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);

  for(int i = 0; i < n_funct; i++){
    splines2::BSpline bspline;
    // Create Bspline object with 8 degrees of freedom
    // 8 - 3 - 1 internal nodes
    bspline = splines2::BSpline(t_obs(i,0), P);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat {bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  arma::mat P_mat(P, P, arma::fill::zeros);
  P_mat.zeros();
  for(int j = 0; j < P_mat.n_rows; j++){
    P_mat(0,0) = 1;
    if(j > 0){
      P_mat(j,j) = 2;
      P_mat(j-1,j) = -1;
      P_mat(j,j-1) = -1;
    }
    P_mat(P_mat.n_rows - 1, P_mat.n_rows - 1) = 1;
  }

  arma::cube nu(K, P, r_stored_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(r_stored_iters);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_funct; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  arma::vec loglik = arma::zeros(r_stored_iters);

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::zeros);
    Phi(i,0) = arma::zeros(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  // Create parameters for tempered transitions using geometric scheme
  arma::vec beta_ladder(N_t, arma::fill::ones);
  beta_ladder(N_t - 1) = beta_N_t;
  double geom_mult = std::pow(beta_N_t, 1.0/N_t);
  Rcpp::Rcout << "geom_mult: " << geom_mult << "\n";
  for(int i = 1; i < N_t; i++){
    beta_ladder(i) = beta_ladder(i-1) * geom_mult;
    // beta_ladder(i) = 1 - ((1- beta_N_t) *(std::pow(i/ (N_t - 1.0), 2.0)));
    Rcpp::Rcout << "beta_i: " << beta_ladder(i) << "\n";
  }
  // Create storage for tempered transitions
  arma::cube nu_TT(K, P, (2 * N_t) + 1, arma::fill::randn);
  arma::cube chi_TT(n_funct, M, (2 * N_t) + 1, arma::fill::zeros);
  arma::mat pi_TT(K, (2 * N_t) + 1, arma::fill::zeros);
  arma::vec sigma_TT((2 * N_t) + 1, arma::fill::ones);
  arma::cube Z_TT = arma::randi<arma::cube>(n_funct, K, (2 * N_t) + 1,
                                            arma::distr_param(0,1));
  arma::mat delta_TT(M, (2 * N_t) + 1, arma::fill::ones);
  arma::field<arma::cube> gamma_TT((2 * N_t) + 1, 1);
  arma::field<arma::cube> Phi_TT((2 * N_t) + 1, 1);
  arma::mat A_TT = arma::ones((2 * N_t) + 1, 2);
  arma::vec alpha_3_TT = arma::ones((2 * N_t) + 1);

  for(int i = 0; i < ((2 * N_t) + 1); i++){
    gamma_TT(i,0) = arma::cube(K, P, M, arma::fill::zeros);
    Phi_TT(i,0) = arma::zeros(K, P, M);
  }

  arma::mat tau_TT((2 * N_t) + 1, K, arma::fill::ones);

  int temp_ind = 0;
  double logA = 0;
  double logu = 0;


  // initialize placeholders
  nu_TT.slice(0) = nu.slice(0);
  chi_TT.slice(0) = chi.slice(0);
  pi_TT.col(0) = pi.col(0);
  sigma_TT(0) = sigma(0);
  Z_TT.slice(0) = Z.slice(0);
  delta_TT.col(0) = delta.col(0);
  gamma_TT(0,0) = gamma(0,0);
  Phi_TT(0,0) = Phi(0,0);
  A_TT.row(0) = A.row(0);
  tau_TT.row(0) = tau.row(0);
  alpha_3_TT(0) = alpha_3(0);

  nu_TT.slice(1) = nu.slice(0);
  chi_TT.slice(1) = chi.slice(0);
  pi_TT.col(1) = pi.col(0);
  sigma_TT(1) = sigma(0);
  Z_TT.slice(1) = Z.slice(0);
  delta_TT.col(1) = delta.col(0);
  gamma_TT(1,0) = gamma(0,0);
  Phi_TT(1,0) = Phi(0,0);
  A_TT.row(1) = A.row(0);
  tau_TT.row(1) = tau.row(0);
  alpha_3_TT(1) = alpha_3_TT(0);

  temp_ind = 0;

  // Perform tempered transitions
  for(int l = 1; l < ((2 * N_t) + 1); l++){
    updateZTempered_PM(beta_ladder(temp_ind), y_obs, B_obs,
                       Phi_TT(l,0), nu_TT.slice(l), chi_TT.slice(l),
                       pi_TT.col(l), sigma_TT(l), l, (2 * N_t) + 1, alpha_3_TT(l), a_Z_PM,
                       Z_ph, Z_TT);
    updatePi_PM(alpha_3_TT(l), Z_TT.slice(l), c, l, (2 * N_t) + 1, a_pi_PM, pi_ph, pi_TT);
    updateAlpha3(pi_TT.col(l), b, Z_TT.slice(l), l, (2 * N_t) + 1, var_alpha3, alpha_3_TT);

    tilde_tau(0) = delta_TT(0, l);
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta_TT(j,l);
    }

    updatePhiTempered(beta_ladder(temp_ind), y_obs, B_obs,
                      nu_TT.slice(l), gamma_TT(l,0), tilde_tau, Z_TT.slice(l),
                      chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, m_1, M_1,
                      Phi_TT);
    updateDelta(Phi_TT(l,0), gamma_TT(l,0), A_TT.row(l).t(), l, (2 * N_t) + 1,
                delta_TT);

    updateA(alpha1l, beta1l, alpha2l, beta2l, delta_TT.col(l), var_epsilon1,
            var_epsilon2, l, (2 * N_t) + 1, A_TT);
    updateGamma(nu_1, delta_TT.col(l), Phi_TT(l,0), l, (2 * N_t) + 1,
                gamma_TT);
    updateNuTempered(beta_ladder(temp_ind), y_obs, B_obs,
                     tau_TT.row(l).t(), Phi_TT(l,0), Z_TT.slice(l),
                     chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, P_mat,
                     b_1, B_1, nu_TT);
    updateTau(alpha, beta, nu_TT.slice(l), l, (2 * N_t) + 1, P_mat, tau_TT);
    updateSigmaTempered(beta_ladder(temp_ind), y_obs, B_obs,
                        alpha_0, beta_0, nu_TT.slice(l), Phi_TT(l,0),
                        Z_TT.slice(l), chi_TT.slice(l), l, (2 * N_t) + 1,
                        sigma_TT);
    updateChiTempered(beta_ladder(temp_ind), y_obs, B_obs,
                      Phi_TT(l,0), nu_TT.slice(l), Z_TT.slice(l), sigma_TT(l),
                      l, (2 * N_t) + 1, chi_TT);

    // update temp_ind
    if(l < N_t){
      temp_ind = temp_ind + 1;
    }
    if(l > N_t){
      temp_ind = temp_ind - 1;
    }


  }

  logA = CalculateTTAcceptance(beta_ladder, y_obs, B_obs,
                               nu_TT, Phi_TT, Z_TT, chi_TT, sigma_TT);
  logu = std::log(R::runif(0,1));

  Rcpp::Rcout << "prob_accept: " << logA<< "\n";
  Rcpp::Rcout << "logu: " << logu<< "\n";

  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu_TT),
                                         Rcpp::Named("alpha_3", alpha_3_TT),
                                         Rcpp::Named("chi", chi_TT),
                                         Rcpp::Named("pi", pi_TT),
                                         Rcpp::Named("A", A_TT),
                                         Rcpp::Named("delta", delta_TT),
                                         Rcpp::Named("sigma", sigma_TT),
                                         Rcpp::Named("tau", tau_TT),
                                         Rcpp::Named("gamma", gamma_TT),
                                         Rcpp::Named("Phi", Phi_TT),
                                         Rcpp::Named("Z", Z_TT));
  return params;
}

// Conducts a mixture of untempered sampling and tempered sampling to get posterior draws from the partial membership model
//
// @name BFPMM_MTT
// @param y_obs Field (list) of vectors containing the observed values
// @param t_obs Field (list) of vectors containing time points of observed values
// @param n_funct Int containing number of functions observed
// @param thinning_num Int containing how often we save an MCMC iteration
// @param K Int containing the number of clusters
// @param basis degree Int containing the degree of B-splines used
// @param M Int containing the number of eigenfunctions
// @param boundary_knots Vector containing the boundary points of our index domain of interest
// @param internal_knots Vector location of internal knots for B-splines
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int constaining number of iterations performed for each batch
// @param rho Double containing hyperparmater for sampling from Z
// @param alpha_3 Double hyperparameter for sampling from pi
// @param a_12 Vec containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param a_Z_PM Double containing hyperparameter used to sample from the posterior of Z
// @param a_pi_PM Double containing hyperparameter used to sample from the posterior of pi
// @param var_alpha3 Double containing hyperparameter for sampling from alpha_3
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param directory String containing path to store batches of MCMC samples
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_MTT(const arma::field<arma::vec>& y_obs,
                            const arma::field<arma::vec>& t_obs,
                            const int& n_funct,
                            const int& thinning_num,
                            const int& K,
                            const int basis_degree,
                            const int& M,
                            const arma::vec boundary_knots,
                            const arma::vec internal_knots,
                            const int& tot_mcmc_iters,
                            const int& r_stored_iters,
                            const int& n_temp_trans,
                            const arma::vec& c,
                            const double& b,
                            const double& nu_1,
                            const double& alpha1l,
                            const double& alpha2l,
                            const double& beta1l,
                            const double& beta2l,
                            const double& a_Z_PM,
                            const double& a_pi_PM,
                            const double& var_alpha3,
                            const double& var_epsilon1,
                            const double& var_epsilon2,
                            const double& alpha,
                            const double& beta,
                            const double& alpha_0,
                            const double& beta_0,
                            const std::string directory,
                            const double& beta_N_t,
                            const int& N_t){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  int P = internal_knots.n_elem + basis_degree + 1;

  for(int i = 0; i < n_funct; i++){
    splines2::BSpline bspline;
    // Create Bspline object
    bspline = splines2::BSpline(t_obs(i,0), internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat {bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  arma::mat P_mat(P, P, arma::fill::zeros);
  P_mat.zeros();
  for(int j = 0; j < P_mat.n_rows; j++){
    P_mat(0,0) = 1;
    if(j > 0){
      P_mat(j,j) = 2;
      P_mat(j-1,j) = -1;
      P_mat(j,j-1) = -1;
    }
    P_mat(P_mat.n_rows - 1, P_mat.n_rows - 1) = 1;
  }

  arma::cube nu(K, P, r_stored_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(r_stored_iters);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_funct; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  // Create parameters for tempered transitions using geometric scheme
  arma::vec beta_ladder(N_t, arma::fill::ones);
  beta_ladder(N_t - 1) = beta_N_t;
  double geom_mult = std::pow(beta_N_t, 1.0/N_t);
  Rcpp::Rcout << "geom_mult: " << geom_mult << "\n";
  for(int i = 1; i < N_t; i++){
    beta_ladder(i) = beta_ladder(i-1) * geom_mult;
    // beta_ladder(i) = 1 - ((1- beta_N_t) *(std::pow(i/ (N_t - 1.0), 2.0)));
    Rcpp::Rcout << "beta_i: " << beta_ladder(i) << "\n";
  }
  // Create storage for tempered transitions
  arma::cube nu_TT(K, P, (2 * N_t) + 1, arma::fill::randn);
  arma::cube chi_TT(n_funct, M, (2 * N_t) + 1, arma::fill::randn);
  arma::mat pi_TT(K, (2 * N_t) + 1, arma::fill::zeros);
  arma::vec sigma_TT((2 * N_t) + 1, arma::fill::ones);
  arma::cube Z_TT = arma::randi<arma::cube>(n_funct, K, (2 * N_t) + 1,
                                            arma::distr_param(0,1));
  arma::mat delta_TT(M, (2 * N_t) + 1, arma::fill::ones);
  arma::field<arma::cube> gamma_TT((2 * N_t) + 1, 1);
  arma::field<arma::cube> Phi_TT((2 * N_t) + 1, 1);
  arma::mat A_TT = arma::ones((2 * N_t) + 1, 2);
  arma::vec alpha_3_TT = arma::ones((2 * N_t) + 1);

  for(int i = 0; i < ((2 * N_t) + 1); i++){
    gamma_TT(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi_TT(i,0) = arma::randn(K, P, M);
  }

  arma::mat tau_TT((2 * N_t) + 1, K, arma::fill::ones);

  int temp_ind = 0;
  double logA = 0;
  double logu = 0;
  int accept_num = 0;

  for(int i=0; i < tot_mcmc_iters; i++){
    if(((i % n_temp_trans) != 0) || (i == 0)){
      updateZ_PM(y_obs, B_obs, Phi((i % r_stored_iters),0),
                 nu.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                 pi.col((i % r_stored_iters)), sigma((i % r_stored_iters)),
                 (i % r_stored_iters), r_stored_iters, alpha_3(i % r_stored_iters),
                 a_Z_PM, Z_ph, Z);

      updatePi_PM(alpha_3(i % r_stored_iters) ,Z.slice(i% r_stored_iters), c,
                  (i % r_stored_iters), r_stored_iters, a_pi_PM, pi_ph, pi);

      updateAlpha3(pi.col(i % r_stored_iters), b, Z.slice(i % r_stored_iters),
                   (i % r_stored_iters), r_stored_iters, var_alpha3, alpha_3);

      tilde_tau(0) = delta(0, (i % r_stored_iters));
      for(int j = 1; j < M; j++){
        tilde_tau(j) = tilde_tau(j-1) * delta(j,(i % r_stored_iters));
      }

      updatePhi(y_obs, B_obs, nu.slice((i % r_stored_iters)),
                gamma((i % r_stored_iters),0), tilde_tau,
                Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                sigma((i % r_stored_iters)), (i % r_stored_iters),
                r_stored_iters, m_1, M_1, Phi);

      updateDelta(Phi((i % r_stored_iters),0), gamma((i % r_stored_iters),0),
                  A.row(i % r_stored_iters).t(), (i % r_stored_iters),
                  r_stored_iters, delta);

      updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i % r_stored_iters)),
              var_epsilon1, var_epsilon2, (i % r_stored_iters), r_stored_iters, A);

      updateGamma(nu_1, delta.col((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                  (i % r_stored_iters), r_stored_iters, gamma);

      updateNu(y_obs, B_obs, tau.row((i % r_stored_iters)).t(),
               Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)),
               chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)),
               (i % r_stored_iters), r_stored_iters, P_mat, b_1, B_1, nu);

      updateTau(alpha, beta, nu.slice((i % r_stored_iters)), (i % r_stored_iters),
                r_stored_iters, P_mat, tau);

      updateSigma(y_obs, B_obs, alpha_0, beta_0,
                  nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                  Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                  (i % r_stored_iters), r_stored_iters, sigma);

      updateChi(y_obs, B_obs, Phi((i % r_stored_iters),0),
                nu.slice((i % r_stored_iters)), Z.slice((i % r_stored_iters)),
                sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters,
                chi);
    }
    if((i % n_temp_trans) == 0 && (i > 0)){
      // initialize placeholders
      nu_TT.slice(0) = nu.slice (i % r_stored_iters);
      chi_TT.slice(0) = chi.slice(i % r_stored_iters);
      pi_TT.col(0) = pi.col(i % r_stored_iters);
      sigma_TT(0) = sigma(i % r_stored_iters);
      Z_TT.slice(0) = Z.slice(i % r_stored_iters);
      delta_TT.col(0) = delta.col(i % r_stored_iters);
      gamma_TT(0,0) = gamma(i % r_stored_iters,0);
      Phi_TT(0,0) = Phi(i % r_stored_iters,0);
      A_TT.row(0) = A.row(i % r_stored_iters);
      tau_TT.row(0) = tau.row(i % r_stored_iters);
      alpha_3_TT(0) = alpha_3(i % r_stored_iters);

      nu_TT.slice(1) = nu.slice(i % r_stored_iters);
      chi_TT.slice(1) = chi.slice(i % r_stored_iters);
      pi_TT.col(1) = pi.col(i % r_stored_iters);
      sigma_TT(1) = sigma(i % r_stored_iters);
      Z_TT.slice(1) = Z.slice(i % r_stored_iters);
      delta_TT.col(1) = delta.col(i % r_stored_iters);
      gamma_TT(1,0) = gamma(i % r_stored_iters,0);
      Phi_TT(1,0) = Phi(i % r_stored_iters,0);
      A_TT.row(1) = A.row(i % r_stored_iters);
      tau_TT.row(1) = tau.row(i % r_stored_iters);
      alpha_3_TT(1) = alpha_3(i % r_stored_iters);

      temp_ind = 0;

      // Perform tempered transitions
      for(int l = 1; l < ((2 * N_t) + 1); l++){
        updateZTempered_PM(beta_ladder(temp_ind), y_obs, B_obs,
                           Phi_TT(l,0), nu_TT.slice(l), chi_TT.slice(l),
                           pi_TT.col(l), sigma_TT(l), l, (2 * N_t) + 1, alpha_3_TT(l), a_Z_PM,
                           Z_ph, Z_TT);
        updatePi_PM(alpha_3_TT(l), Z_TT.slice(l), c, l, (2 * N_t) + 1, a_pi_PM, pi_ph, pi_TT);
        updateAlpha3(pi_TT.col(l), b, Z_TT.slice(l), l, (2 * N_t) + 1, var_alpha3, alpha_3_TT);

        tilde_tau(0) = delta_TT(0, l);
        for(int j = 1; j < M; j++){
          tilde_tau(j) = tilde_tau(j-1) * delta_TT(j,l);
        }

        updatePhiTempered(beta_ladder(temp_ind), y_obs, B_obs,
                          nu_TT.slice(l), gamma_TT(l,0), tilde_tau, Z_TT.slice(l),
                          chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, m_1, M_1,
                          Phi_TT);
        updateDelta(Phi_TT(l,0), gamma_TT(l,0), A_TT.row(l).t(), l, (2 * N_t) + 1,
                    delta_TT);

        updateA(alpha1l, beta1l, alpha2l, beta2l, delta_TT.col(l), var_epsilon1,
                var_epsilon2, l, (2 * N_t) + 1, A_TT);
        updateGamma(nu_1, delta_TT.col(l), Phi_TT(l,0), l, (2 * N_t) + 1,
                    gamma_TT);
        updateNuTempered(beta_ladder(temp_ind), y_obs, B_obs,
                         tau_TT.row(l).t(), Phi_TT(l,0), Z_TT.slice(l),
                         chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, P_mat,
                         b_1, B_1, nu_TT);
        updateTau(alpha, beta, nu_TT.slice(l), l, (2 * N_t) + 1, P_mat, tau_TT);
        updateSigmaTempered(beta_ladder(temp_ind), y_obs, B_obs,
                            alpha_0, beta_0, nu_TT.slice(l), Phi_TT(l,0),
                            Z_TT.slice(l), chi_TT.slice(l), l, (2 * N_t) + 1,
                            sigma_TT);
        updateChiTempered(beta_ladder(temp_ind), y_obs, B_obs,
                          Phi_TT(l,0), nu_TT.slice(l), Z_TT.slice(l), sigma_TT(l),
                          l, (2 * N_t) + 1, chi_TT);

        // update temp_ind
        if(l < N_t){
          temp_ind = temp_ind + 1;
        }
        if(l > N_t){
          temp_ind = temp_ind - 1;
        }
      }
      logA = CalculateTTAcceptance(beta_ladder, y_obs, B_obs,
                                   nu_TT, Phi_TT, Z_TT, chi_TT, sigma_TT);
      logu = std::log(R::runif(0,1));

      Rcpp::Rcout << "prob_accept: " << logA<< "\n";
      Rcpp::Rcout << "logu: " << logu<< "\n";

      if(logu < logA){
        Rcpp::Rcout << "Accept \n";
        nu.slice(i % r_stored_iters) = nu_TT.slice(2 * N_t);
        chi.slice(i % r_stored_iters) = chi_TT.slice(2 * N_t);
        pi.col(i % r_stored_iters) = pi_TT.col(2 * N_t);
        sigma(i % r_stored_iters) = sigma_TT(2 * N_t);
        Z.slice(i % r_stored_iters) = Z_TT.slice(2 * N_t);
        delta.col(i % r_stored_iters) = delta_TT.col(2 * N_t);
        gamma(i % r_stored_iters,0) = gamma_TT(2 * N_t,0);
        Phi(i % r_stored_iters,0) = Phi_TT(2 * N_t,0);
        A.row(i % r_stored_iters) = A_TT.row(2 * N_t);
        tau.row(i % r_stored_iters) = tau_TT.row(2 * N_t);
        alpha_3(i % r_stored_iters) = alpha_3_TT(2 * N_t);

        //update accept number
        accept_num = accept_num + 1;
      }

      //initialize next state
      if(((i+1) % r_stored_iters) != 0){
        nu.slice((i+1) % r_stored_iters) = nu.slice(i % r_stored_iters);
        chi.slice((i+1) % r_stored_iters) = chi.slice(i % r_stored_iters);
        pi.col((i+1) % r_stored_iters) = pi.col(i % r_stored_iters);
        sigma((i+1) % r_stored_iters) = sigma(i % r_stored_iters);
        Z.slice((i+1) % r_stored_iters) = Z.slice(i % r_stored_iters);
        delta.col((i+1) % r_stored_iters) = delta.col(i % r_stored_iters);
        A.row((i+1) % r_stored_iters) = A.row(i % r_stored_iters);
        tau.row((i+1) % r_stored_iters) = tau.row(i % r_stored_iters);
        Phi((i+1) % r_stored_iters,0) = Phi(i % r_stored_iters, 0);
        alpha_3((i+1) % r_stored_iters) =  alpha_3(i % r_stored_iters);
      }
    }
    loglik((i % r_stored_iters)) =  calcLikelihood(y_obs, B_obs,
           nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
           Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
           sigma((i % r_stored_iters)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Accpetance Probability: " << accept_num / (std::round(i / n_temp_trans)) << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i % r_stored_iters)-4, (i % r_stored_iters))) << "\n";
      Rcpp::checkUserInterrupt();
    }
    if(((i+1) % r_stored_iters) == 0 && i > 1){
      // Save parameters
      arma::cube nu1(K, P, r_stored_iters/thinning_num, arma::fill::randn);
      arma::cube chi1(n_funct, M, r_stored_iters/thinning_num, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec alpha_31(r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters/thinning_num, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters/thinning_num, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_funct, K, r_stored_iters/thinning_num,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters/thinning_num, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters/thinning_num,1);
      arma::field<arma::cube> Phi1(r_stored_iters/thinning_num, 1);
      arma::mat tau1(r_stored_iters/thinning_num, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);

      for(int p=1; p < r_stored_iters / thinning_num; p++){
        nu1.slice(p) = nu.slice(thinning_num*p - 1);
        chi1.slice(p) = chi.slice(thinning_num*p - 1);
        pi1.col(p) = pi.col(thinning_num*p - 1);
        alpha_31(p) = alpha_3(thinning_num*p - 1);
        sigma1(p) = sigma(thinning_num*p - 1);
        A1.row(p) = A.row(thinning_num*p - 1);
        Z1.slice(p) = Z.slice(thinning_num*p - 1);
        delta1.col(p) = delta.col(thinning_num*p - 1);
        gamma1(p,0) = gamma(thinning_num*p - 1,0);
        Phi1(p,0) = Phi(thinning_num*p  - 1,0);
        tau1.row(p) = tau.row(thinning_num*p - 1);

        nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
        chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
        pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
        alpha_31.save(directory + "alpha_3" + std::to_string(q) + ".txt", arma::arma_ascii);
        A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
        delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
        sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
        tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
        gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
        Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
        Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

        //reset all parameters
        nu.slice(0) = nu.slice(i % r_stored_iters);
        chi.slice(0) = chi.slice(i % r_stored_iters);
        pi.col(0) = pi.col(i % r_stored_iters);
        alpha_3(0) = alpha_3(i % r_stored_iters);
        A.row(0) = A.row(i % r_stored_iters);
        delta.col(0) = delta.col(i % r_stored_iters);
        sigma(0) = sigma(i % r_stored_iters);
        tau.row(0) = tau.row(i % r_stored_iters);
        gamma(0,0) = gamma(i % r_stored_iters, 0);
        Phi(0,0) = Phi(i % r_stored_iters, 0);
        Z.slice(0) = Z.slice(i % r_stored_iters);

        q = q + 1;
      }
    }
  }

  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu_TT),
                                         Rcpp::Named("alpha_3", alpha_3_TT),
                                         Rcpp::Named("chi", chi_TT),
                                         Rcpp::Named("pi", pi_TT),
                                         Rcpp::Named("A", A_TT),
                                         Rcpp::Named("delta", delta_TT),
                                         Rcpp::Named("sigma", sigma_TT),
                                         Rcpp::Named("tau", tau_TT),
                                         Rcpp::Named("gamma", gamma_TT),
                                         Rcpp::Named("Phi", Phi_TT),
                                         Rcpp::Named("Z", Z_TT),
                                         Rcpp::Named("loglik", loglik));
  return params;
}

// Conducts un-tempered MCMC to mean and allocation parameters
//
// @name BFPMM_Nu_Z
// @param y_obs Field (list) of vectors containing the observed values
// @param t_obs Field (list) of vectors containing time points of observed values
// @param n_funct Int containing number of functions observed
// @param K Int containing the number of clusters
// @param basis degree Int containing the degree of B-splines used
// @param M Int containing the number of eigenfunctions
// @param boundary_knots Vector containing the boundary points of our index domain of interest
// @param internal_knots Vector location of internal knots for B-splines
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int containing number of iterations performed for each batch
// @param c Vector containing hyperparmeters for pi
// @param b double containing hyperparameter for alpha_3
// @param a_12 Vector containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param var_pi Double containing variance parameter of the random walk MH for pi parameter
// @param var_Z Double containing variance parameter of the random walk MH for Z parameter
// @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_Nu_Z(const arma::field<arma::vec>& y_obs,
                             const arma::field<arma::vec>& t_obs,
                             const int& n_funct,
                             const int& K,
                             const int basis_degree,
                             const int& M,
                             const arma::vec boundary_knots,
                             const arma::vec internal_knots,
                             const int& tot_mcmc_iters,
                             const arma::vec& c,
                             const double& b,
                             const double& alpha1l,
                             const double& alpha2l,
                             const double& beta1l,
                             const double& beta2l,
                             const double& a_Z_PM,
                             const double& a_pi_PM,
                             const double& var_alpha3,
                             const double& var_epsilon1,
                             const double& var_epsilon2,
                             const double& alpha,
                             const double& beta,
                             const double& alpha_0,
                             const double& beta_0){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  int P = internal_knots.n_elem + basis_degree + 1;

  for(int i = 0; i < n_funct; i++){
    splines2::BSpline bspline;
    // Create Bspline object
    bspline = splines2::BSpline(t_obs(i,0), internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat {bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  arma::mat P_mat(P, P, arma::fill::zeros);
  P_mat.zeros();
  for(int j = 0; j < P_mat.n_rows; j++){
    P_mat(0,0) = 1;
    if(j > 0){
      P_mat(j,j) = 2;
      P_mat(j-1,j) = -1;
      P_mat(j,j-1) = -1;
    }
    P_mat(P_mat.n_rows - 1, P_mat.n_rows - 1) = 1;
  }

  arma::cube nu(K, P, tot_mcmc_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, tot_mcmc_iters, arma::fill::zeros);
  arma::mat pi(K, tot_mcmc_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(tot_mcmc_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(tot_mcmc_iters);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, tot_mcmc_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_funct; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, tot_mcmc_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(tot_mcmc_iters,1);
  arma::field<arma::cube> Phi(tot_mcmc_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(tot_mcmc_iters, 2);
  arma::vec loglik = arma::zeros(tot_mcmc_iters);

  for(int i = 0; i < tot_mcmc_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::zeros(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(tot_mcmc_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  for(int i = 0; i < tot_mcmc_iters; i++){
    updateZ_PM(y_obs, B_obs, Phi(i,0),
               nu.slice(i), chi.slice(i),
               pi.col(i), sigma(i),
               i, tot_mcmc_iters, alpha_3(i),
               a_Z_PM, Z_ph, Z);
    updatePi_PM(alpha_3(i) ,Z.slice(i), c,
                (i), tot_mcmc_iters, a_pi_PM, pi_ph, pi);

    updateAlpha3(pi.col(i), b, Z.slice(i),
                 (i), tot_mcmc_iters, var_alpha3, alpha_3);

    tilde_tau(0) = delta(0, (i));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i));
    }

    updateNu(y_obs, B_obs, tau.row((i)).t(),
             Phi((i),0), Z.slice((i)),
             chi.slice((i)), sigma((i)),
             (i), tot_mcmc_iters, P_mat, b_1, B_1, nu);

    updateTau(alpha, beta, nu.slice((i)), (i),
              tot_mcmc_iters, P_mat, tau);

    updateSigma(y_obs, B_obs, alpha_0, beta_0,
                nu.slice((i)), Phi((i),0),
                Z.slice((i)), chi.slice((i)),
                (i), tot_mcmc_iters, sigma);

    // Calculate log likelihood
    loglik((i)) =  calcLikelihood(y_obs, B_obs, nu.slice((i)),
           Phi((i),0), Z.slice((i)), chi.slice((i)), sigma((i)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i)-19, (i))) << "\n";
      Rcpp::checkUserInterrupt();
    }
  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
                                         Rcpp::Named("pi", pi),
                                         Rcpp::Named("alpha_3", alpha_3),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("Z", Z),
                                         Rcpp::Named("loglik", loglik));
  return params;
}


// Conducts un-tempered MCMC to estimate the posterior distribution of parameters not related to Z or Nu, conditioned on a value of Nu and Z
//
// @name BFPMM_Theta
// @param y_obs Field (list) of vectors containing the observed values
// @param t_obs Field (list) of vectors containing time points of observed values
// @param n_funct Int containing number of functions observed
// @param K Int containing the number of clusters
// @param basis degree Int containing the degree of B-splines used
// @param M Int containing the number of eigenfunctions
// @param boundary_knots Vector containing the boundary points of our index domain of interest
// @param internal_knots Vector location of internal knots for B-splines
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param c Vector containing hyperparmeter for pi
// @param b double containing hyperparamete for alpha_3
// @param alpha1l Double containing hyperparameter for sampling from A
// @param alpha2l Double containing hyperparameter for sampling from A
// @param beta1l Double containing hyperparameter for sampling from A
// @param beta2l Double containing hyperparameter for sampling from A
// @param var_pi Double containing variance parameter of the random walk MH for pi parameter
// @param var_Z Double containing variance parameter of the random walk MH for Z parameter
// @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
// @param var_epslion1 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameter for sampling from tau
// @param beta Double containing hyperparameter for sampling from tau
// @param alpha_0 Double containing hyperparameter for sampling from sigma
// @param beta_0 Double containing hyperparameter for sampling from sigma
// @param Z_est Matrix containing Z values to be conditioned on
// @param nu_est Matrix containing nu values to be conditioned on
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_Theta(const arma::field<arma::vec>& y_obs,
                              const arma::field<arma::vec>& t_obs,
                              const int& n_funct,
                              const int& K,
                              const int basis_degree,
                              const int& M,
                              const arma::vec boundary_knots,
                              const arma::vec internal_knots,
                              const int& tot_mcmc_iters,
                              const arma::vec& c,
                              const double& b,
                              const double& nu_1,
                              const double& alpha1l,
                              const double& alpha2l,
                              const double& beta1l,
                              const double& beta2l,
                              const double& a_Z_PM,
                              const double& a_pi_PM,
                              const double& var_alpha3,
                              const double& var_epsilon1,
                              const double& var_epsilon2,
                              const double& alpha,
                              const double& beta,
                              const double& alpha_0,
                              const double& beta_0,
                              const arma::mat& Z_est,
                              const arma::mat& nu_est){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  int P = internal_knots.n_elem + basis_degree + 1;

  for(int i = 0; i < n_funct; i++){
    splines2::BSpline bspline;
    // Create Bspline object
    bspline = splines2::BSpline(t_obs(i,0), internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat {bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  arma::mat P_mat(P, P, arma::fill::zeros);
  P_mat.zeros();
  for(int j = 0; j < P_mat.n_rows; j++){
    P_mat(0,0) = 1;
    if(j > 0){
      P_mat(j,j) = 2;
      P_mat(j-1,j) = -1;
      P_mat(j,j-1) = -1;
    }
    P_mat(P_mat.n_rows - 1, P_mat.n_rows - 1) = 1;
  }

  arma::cube nu(K, P, tot_mcmc_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, tot_mcmc_iters, arma::fill::randn);
  arma::mat pi(K, tot_mcmc_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(tot_mcmc_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(tot_mcmc_iters);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, tot_mcmc_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_funct; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, tot_mcmc_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(tot_mcmc_iters,1);
  arma::field<arma::cube> Phi(tot_mcmc_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(tot_mcmc_iters, 2);
  arma::vec loglik = arma::zeros(tot_mcmc_iters);

  for(int i = 0; i < tot_mcmc_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(tot_mcmc_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  Z.slice(0) = Z_est;
  nu.slice(0) = nu_est;

  for(int i = 1; i < tot_mcmc_iters; i++){
    Z.slice(i) = Z_est;
    nu.slice(i) = nu_est;
  }


  for(int i = 0; i < tot_mcmc_iters; i++){
    tilde_tau(0) = delta(0, (i));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i));
    }

    updatePhi(y_obs, B_obs, nu.slice((i)),
              gamma((i),0), tilde_tau,
              Z.slice((i)), chi.slice((i)),
              sigma((i)), (i),
              tot_mcmc_iters, m_1, M_1, Phi);

    updateDelta(Phi((i),0), gamma((i),0),
                A.row(i).t(), (i),
                tot_mcmc_iters, delta);

    updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i)),
            var_epsilon1, var_epsilon2, (i), tot_mcmc_iters, A);

    updateGamma(nu_1, delta.col((i)), Phi((i),0),
                (i), tot_mcmc_iters, gamma);

    updateTau(alpha, beta, nu.slice((i)), (i),
              tot_mcmc_iters, P_mat, tau);

    updateSigma(y_obs, B_obs, alpha_0, beta_0,
                nu.slice((i)), Phi((i),0),
                Z.slice((i)), chi.slice((i)),
                (i), tot_mcmc_iters, sigma);

    updateChi(y_obs, B_obs, Phi((i),0),
              nu.slice((i)), Z.slice((i)),
              sigma((i)), (i), tot_mcmc_iters,
              chi);

    // Calculate log likelihood
    loglik((i)) =  calcLikelihood(y_obs, B_obs, nu.slice((i)),
           Phi((i),0), Z.slice((i)), chi.slice((i)), sigma((i)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i)-99, (i))) << "\n";
      Rcpp::checkUserInterrupt();
    }
  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("Z", Z),
                                         Rcpp::Named("nu", nu),
                                         Rcpp::Named("chi", chi),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("gamma", gamma),
                                         Rcpp::Named("Phi", Phi),
                                         Rcpp::Named("loglik", loglik));
  return params;
}


// Conducts a mixture of untempered sampling and termpered sampling to get posterior draws from the partial membership model
//
// @name BFPMM_MTT_warm_start
// @param y_obs Field (list) of vectors containing the observed values
// @param t_obs Field (list) of vectors containing time points of observed values
// @param n_funct Int containing number of functions observed
// @param thinning_num Int containing how often we save an MCMC iteration
// @param K Int containing the number of clusters
// @param basis degree Int containing the degree of B-splines used
// @param M Int containing the number of eigenfunctions
// @param boundary_knots Vector containing the boundary points of our index domain of interest
// @param internal_knots Vector location of internal knots for B-splines
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int constaining number of iterations performed for each batch
// @param t_star Field (list) of vectors containing time points of interest that are not observed (optional)
// @param rho Double containing hyperparmater for sampling from Z
// @param alpha_3 Double hyperparameter for sampling from pi
// @param a_12 Vec containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param a_Z_PM Double containing hyperparameter used to sample from the posterior of Z
// @param a_pi_PM Double containing hyperparameter used to sample from the posterior of pi
// @param var_alpha3 Doubel containing hyperparameter for sampling from alpha_3
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param directory String containing path to store batches of MCMC samples
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_MTT_warm_start(const arma::field<arma::vec>& y_obs,
                                       const arma::field<arma::vec>& t_obs,
                                       const int& n_funct,
                                       const int& thinning_num,
                                       const int& K,
                                       const int basis_degree,
                                       const int& M,
                                       const arma::vec boundary_knots,
                                       const arma::vec internal_knots,
                                       const int& tot_mcmc_iters,
                                       const int& r_stored_iters,
                                       const int& n_temp_trans,
                                       const arma::vec& c,
                                       const double& b,
                                       const double& nu_1,
                                       const double& alpha1l,
                                       const double& alpha2l,
                                       const double& beta1l,
                                       const double& beta2l,
                                       const double& a_Z_PM,
                                       const double& a_pi_PM,
                                       const double& var_alpha3,
                                       const double& var_epsilon1,
                                       const double& var_epsilon2,
                                       const double& alpha,
                                       const double& beta,
                                       const double& alpha_0,
                                       const double& beta_0,
                                       const std::string directory,
                                       const double& beta_N_t,
                                       const int& N_t,
                                       const arma::mat& Z_est,
                                       const arma::vec& pi_est,
                                       const double& alpha_3_est,
                                       const arma::vec& delta_est,
                                       const arma::cube& gamma_est,
                                       const arma::cube& Phi_est,
                                       const arma::vec& A_est,
                                       const arma::mat& nu_est,
                                       const arma::vec& tau_est,
                                       const double& sigma_est,
                                       const arma::mat& chi_est){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  int P = internal_knots.n_elem + basis_degree + 1;

  for(int i = 0; i < n_funct; i++){
    splines2::BSpline bspline;
    // Create Bspline object
    bspline = splines2::BSpline(t_obs(i,0), internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat {bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  arma::mat P_mat(P, P, arma::fill::zeros);
  P_mat.zeros();
  for(int j = 0; j < P_mat.n_rows; j++){
    P_mat(0,0) = 1;
    if(j > 0){
      P_mat(j,j) = 2;
      P_mat(j-1,j) = -1;
      P_mat(j,j-1) = -1;
    }
    P_mat(P_mat.n_rows - 1, P_mat.n_rows - 1) = 1;
  }

  arma::cube nu(K, P, r_stored_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(r_stored_iters);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_funct; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  // Create parameters for tempered transitions using geometric scheme
  arma::vec beta_ladder(N_t, arma::fill::ones);
  beta_ladder(N_t - 1) = beta_N_t;
  double geom_mult = std::pow(beta_N_t, 1.0/N_t);
  // Rcpp::Rcout << "geom_mult: " << geom_mult << "\n";
  for(int i = 1; i < N_t; i++){
    beta_ladder(i) = beta_ladder(i-1) * geom_mult;
    // beta_ladder(i) = 1 - ((1- beta_N_t) *(std::pow(i/ (N_t - 1.0), 2.0)));
    // Rcpp::Rcout << "beta_i: " << beta_ladder(i) << "\n";
  }
  // Create storage for tempered transitions
  arma::cube nu_TT(K, P, (2 * N_t) + 1, arma::fill::randn);
  arma::cube chi_TT(n_funct, M, (2 * N_t) + 1, arma::fill::randn);
  arma::mat pi_TT(K, (2 * N_t) + 1, arma::fill::zeros);
  arma::vec sigma_TT((2 * N_t) + 1, arma::fill::ones);
  arma::cube Z_TT = arma::randi<arma::cube>(n_funct, K, (2 * N_t) + 1,
                                            arma::distr_param(0,1));
  arma::mat delta_TT(M, (2 * N_t) + 1, arma::fill::ones);
  arma::field<arma::cube> gamma_TT((2 * N_t) + 1, 1);
  arma::field<arma::cube> Phi_TT((2 * N_t) + 1, 1);
  arma::mat A_TT = arma::ones((2 * N_t) + 1, 2);
  arma::vec alpha_3_TT = arma::ones((2 * N_t) + 1);

  for(int i = 0; i < ((2 * N_t) + 1); i++){
    gamma_TT(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi_TT(i,0) = arma::randn(K, P, M);
  }

  arma::mat tau_TT((2 * N_t) + 1, K, arma::fill::ones);

  int temp_ind = 0;
  double logA = 0;
  double logu = 0;
  int accept_num = 0;

  Z.slice(0) = Z_est;;
  pi.col(0) = pi_est;

  alpha_3(0) = alpha_3_est;
  delta.col(0) = delta_est;
  gamma(0,0) = gamma_est;
  Phi(0,0) = Phi_est;
  A.row(0) = A_est.t();
  nu.slice(0) = nu_est;
  tau.row(0) = tau_est.t();
  sigma(0) = sigma_est;

  chi.slice(0) = chi_est;

  for(int i=0; i < tot_mcmc_iters; i++){
    if(((i % n_temp_trans) != 0) || (i == 0)){
      updateZ_PM(y_obs, B_obs, Phi((i % r_stored_iters),0),
                 nu.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                 pi.col((i % r_stored_iters)), sigma((i % r_stored_iters)),
                 (i % r_stored_iters), r_stored_iters, alpha_3(i % r_stored_iters),
                 a_Z_PM, Z_ph, Z);

      updatePi_PM(alpha_3(i % r_stored_iters) ,Z.slice(i% r_stored_iters), c,
                  (i % r_stored_iters), r_stored_iters, a_pi_PM, pi_ph, pi);

      updateAlpha3(pi.col(i % r_stored_iters), b, Z.slice(i % r_stored_iters),
                   (i % r_stored_iters), r_stored_iters, var_alpha3, alpha_3);

      tilde_tau(0) = delta(0, (i % r_stored_iters));
      for(int j = 1; j < M; j++){
        tilde_tau(j) = tilde_tau(j-1) * delta(j,(i % r_stored_iters));
      }

      updatePhi(y_obs, B_obs, nu.slice((i % r_stored_iters)),
                gamma((i % r_stored_iters),0), tilde_tau,
                Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                sigma((i % r_stored_iters)), (i % r_stored_iters),
                r_stored_iters, m_1, M_1, Phi);

      updateDelta(Phi((i % r_stored_iters),0), gamma((i % r_stored_iters),0),
                  A.row(i % r_stored_iters).t(), (i % r_stored_iters),
                  r_stored_iters, delta);

      updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i % r_stored_iters)),
              var_epsilon1, var_epsilon2, (i % r_stored_iters), r_stored_iters, A);

      updateGamma(nu_1, delta.col((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                  (i % r_stored_iters), r_stored_iters, gamma);

      updateNu(y_obs, B_obs, tau.row((i % r_stored_iters)).t(),
               Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)),
               chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)),
               (i % r_stored_iters), r_stored_iters, P_mat, b_1, B_1, nu);

      updateTau(alpha, beta, nu.slice((i % r_stored_iters)), (i % r_stored_iters),
                r_stored_iters, P_mat, tau);

      updateSigma(y_obs, B_obs, alpha_0, beta_0,
                  nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                  Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                  (i % r_stored_iters), r_stored_iters, sigma);

      updateChi(y_obs, B_obs, Phi((i % r_stored_iters),0),
                nu.slice((i % r_stored_iters)), Z.slice((i % r_stored_iters)),
                sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters,
                chi);
    }

    if((i % n_temp_trans) == 0 && (i > 0)){
      // initialize placeholders
      nu_TT.slice(0) = nu.slice (i % r_stored_iters);
      chi_TT.slice(0) = chi.slice(i % r_stored_iters);
      pi_TT.col(0) = pi.col(i % r_stored_iters);
      sigma_TT(0) = sigma(i % r_stored_iters);
      Z_TT.slice(0) = Z.slice(i % r_stored_iters);
      delta_TT.col(0) = delta.col(i % r_stored_iters);
      gamma_TT(0,0) = gamma(i % r_stored_iters,0);
      Phi_TT(0,0) = Phi(i % r_stored_iters,0);
      A_TT.row(0) = A.row(i % r_stored_iters);
      tau_TT.row(0) = tau.row(i % r_stored_iters);
      alpha_3_TT(0) = alpha_3(i % r_stored_iters);

      nu_TT.slice(1) = nu.slice(i % r_stored_iters);
      chi_TT.slice(1) = chi.slice(i % r_stored_iters);
      pi_TT.col(1) = pi.col(i % r_stored_iters);
      sigma_TT(1) = sigma(i % r_stored_iters);
      Z_TT.slice(1) = Z.slice(i % r_stored_iters);
      delta_TT.col(1) = delta.col(i % r_stored_iters);
      gamma_TT(1,0) = gamma(i % r_stored_iters,0);
      Phi_TT(1,0) = Phi(i % r_stored_iters,0);
      A_TT.row(1) = A.row(i % r_stored_iters);
      tau_TT.row(1) = tau.row(i % r_stored_iters);
      alpha_3_TT(1) = alpha_3(i % r_stored_iters);

      temp_ind = 0;

      // Perform tempered transitions
      for(int l = 1; l < ((2 * N_t) + 1); l++){
        updateZTempered_PM(beta_ladder(temp_ind), y_obs, B_obs,
                           Phi_TT(l,0), nu_TT.slice(l), chi_TT.slice(l),
                           pi_TT.col(l), sigma_TT(l), l, (2 * N_t) + 1, alpha_3_TT(l), a_Z_PM,
                           Z_ph, Z_TT);
        updatePi_PM(alpha_3_TT(l), Z_TT.slice(l), c, l, (2 * N_t) + 1, a_pi_PM, pi_ph, pi_TT);
        updateAlpha3(pi_TT.col(l), b, Z_TT.slice(l), l, (2 * N_t) + 1, var_alpha3, alpha_3_TT);

        tilde_tau(0) = delta_TT(0, l);
        for(int j = 1; j < M; j++){
          tilde_tau(j) = tilde_tau(j-1) * delta_TT(j,l);
        }

        updatePhiTempered(beta_ladder(temp_ind), y_obs, B_obs,
                          nu_TT.slice(l), gamma_TT(l,0), tilde_tau, Z_TT.slice(l),
                          chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, m_1, M_1,
                          Phi_TT);
        updateDelta(Phi_TT(l,0), gamma_TT(l,0), A_TT.row(l).t(), l, (2 * N_t) + 1,
                    delta_TT);

        updateA(alpha1l, beta1l, alpha2l, beta2l, delta_TT.col(l), var_epsilon1,
                var_epsilon2, l, (2 * N_t) + 1, A_TT);
        updateGamma(nu_1, delta_TT.col(l), Phi_TT(l,0), l, (2 * N_t) + 1,
                    gamma_TT);
        updateNuTempered(beta_ladder(temp_ind), y_obs, B_obs,
                         tau_TT.row(l).t(), Phi_TT(l,0), Z_TT.slice(l),
                         chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, P_mat,
                         b_1, B_1, nu_TT);
        updateTau(alpha, beta, nu_TT.slice(l), l, (2 * N_t) + 1, P_mat, tau_TT);
        updateSigmaTempered(beta_ladder(temp_ind), y_obs, B_obs,
                            alpha_0, beta_0, nu_TT.slice(l), Phi_TT(l,0),
                            Z_TT.slice(l), chi_TT.slice(l), l, (2 * N_t) + 1,
                            sigma_TT);
        updateChiTempered(beta_ladder(temp_ind), y_obs, B_obs,
                          Phi_TT(l,0), nu_TT.slice(l), Z_TT.slice(l), sigma_TT(l),
                          l, (2 * N_t) + 1, chi_TT);
        // update temp_ind
        if(l < N_t){
          temp_ind = temp_ind + 1;
        }
        if(l > N_t){
          temp_ind = temp_ind - 1;
        }
      }
      logA = CalculateTTAcceptance(beta_ladder, y_obs, B_obs,
                                   nu_TT, Phi_TT, Z_TT, chi_TT, sigma_TT);
      logu = std::log(R::runif(0,1));

      Rcpp::Rcout << "prob_accept: " << logA<< "\n";
      Rcpp::Rcout << "logu: " << logu<< "\n";

      if(logu < logA){
        Rcpp::Rcout << "Accept \n";
        nu.slice(i % r_stored_iters) = nu_TT.slice(2 * N_t);
        chi.slice(i % r_stored_iters) = chi_TT.slice(2 * N_t);
        pi.col(i % r_stored_iters) = pi_TT.col(2 * N_t);
        sigma(i % r_stored_iters) = sigma_TT(2 * N_t);
        Z.slice(i % r_stored_iters) = Z_TT.slice(2 * N_t);
        delta.col(i % r_stored_iters) = delta_TT.col(2 * N_t);
        gamma(i % r_stored_iters,0) = gamma_TT(2 * N_t,0);
        Phi(i % r_stored_iters,0) = Phi_TT(2 * N_t,0);
        A.row(i % r_stored_iters) = A_TT.row(2 * N_t);
        tau.row(i % r_stored_iters) = tau_TT.row(2 * N_t);
        alpha_3(i % r_stored_iters) = alpha_3_TT(2 * N_t);

        //update accept number
        accept_num = accept_num + 1;
      }

      //initialize next state
      if(((i+1) % r_stored_iters) != 0){
        nu.slice((i+1) % r_stored_iters) = nu.slice(i % r_stored_iters);
        chi.slice((i+1) % r_stored_iters) = chi.slice(i % r_stored_iters);
        pi.col((i+1) % r_stored_iters) = pi.col(i % r_stored_iters);
        sigma((i+1) % r_stored_iters) = sigma(i % r_stored_iters);
        Z.slice((i+1) % r_stored_iters) = Z.slice(i % r_stored_iters);
        delta.col((i+1) % r_stored_iters) = delta.col(i % r_stored_iters);
        A.row((i+1) % r_stored_iters) = A.row(i % r_stored_iters);
        tau.row((i+1) % r_stored_iters) = tau.row(i % r_stored_iters);
        Phi((i+1) % r_stored_iters,0) = Phi(i % r_stored_iters, 0);
        alpha_3((i+1) % r_stored_iters) =  alpha_3(i % r_stored_iters);
      }
    }
    loglik((i % r_stored_iters)) =  calcLikelihood(y_obs, B_obs,
           nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
           Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
           sigma((i % r_stored_iters)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Accpetance Probability: " << accept_num / (std::round(i / n_temp_trans)) << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i % r_stored_iters)-4, (i % r_stored_iters))) << "\n";
      Rcpp::checkUserInterrupt();
    }
    if(((i+1) % r_stored_iters) == 0 && i > 1){
      // Save parameters
      arma::cube nu1(K, P, r_stored_iters/thinning_num, arma::fill::randn);
      arma::cube chi1(n_funct, M, r_stored_iters/thinning_num, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec alpha_31(r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters/thinning_num, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters/thinning_num, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_funct, K, r_stored_iters/thinning_num,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters/thinning_num, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters/thinning_num,1);
      arma::field<arma::cube> Phi1(r_stored_iters/thinning_num, 1);
      arma::mat tau1(r_stored_iters/thinning_num, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);

      for(int p=1; p < r_stored_iters / thinning_num; p++){
        nu1.slice(p) = nu.slice(thinning_num*p - 1);
        chi1.slice(p) = chi.slice(thinning_num*p - 1);
        pi1.col(p) = pi.col(thinning_num*p - 1);
        alpha_31(p) = alpha_3(thinning_num*p - 1);
        sigma1(p) = sigma(thinning_num*p - 1);
        A1.row(p) = A.row(thinning_num*p - 1);
        Z1.slice(p) = Z.slice(thinning_num*p - 1);
        delta1.col(p) = delta.col(thinning_num*p - 1);
        gamma1(p,0) = gamma(thinning_num*p - 1,0);
        Phi1(p,0) = Phi(thinning_num*p  - 1,0);
        tau1.row(p) = tau.row(thinning_num*p - 1);
      }

      nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
      chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
      pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
      alpha_31.save(directory + "alpha_3" + std::to_string(q) + ".txt", arma::arma_ascii);
      A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
      delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
      sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
      tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
      gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
      Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
      Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

      //reset all parameters
      nu.slice(0) = nu.slice(i % r_stored_iters);
      chi.slice(0) = chi.slice(i % r_stored_iters);
      pi.col(0) = pi.col(i % r_stored_iters);
      alpha_3(0) = alpha_3(i % r_stored_iters);
      A.row(0) = A.row(i % r_stored_iters);
      delta.col(0) = delta.col(i % r_stored_iters);
      sigma(0) = sigma(i % r_stored_iters);
      tau.row(0) = tau.row(i % r_stored_iters);
      gamma(0,0) = gamma(i % r_stored_iters, 0);
      Phi(0,0) = Phi(i % r_stored_iters, 0);
      Z.slice(0) = Z.slice(i % r_stored_iters);

      q = q + 1;
    }
  }

  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
                                         Rcpp::Named("alpha_3", alpha_3),
                                         Rcpp::Named("chi", chi),
                                         Rcpp::Named("pi", pi),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("gamma", gamma),
                                         Rcpp::Named("Phi", Phi),
                                         Rcpp::Named("Z", Z),
                                         Rcpp::Named("loglik", loglik));
  return params;
}


// Conducts a mixture of untempered sampling and tempered sampling to get posterior draws from the multivariate partial membership model
//
// @name BFPMM_MTTMV
// @param y_obs Matrix of observed vectors
// @param thinning_num Int containing how often we save an MCMC iteration
// @param K Int containing the number of clusters
// @param M Int containing the number of eigenfunctions
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int constaining number of iterations performed for each batch
// @param rho Double containing hyperparmater for sampling from Z
// @param alpha_3 Double hyperparameter for sampling from pi
// @param a_12 Vec containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param a_Z_PM Double containing hyperparameter used to sample from the posterior of Z
// @param a_pi_PM Double containing hyperparameter used to sample from the posterior of pi
// @param var_alpha3 Doubel containing hyperparameter for sampling from alpha_3
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param directory String containing path to store batches of MCMC samples
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_MTTMV(const arma::mat& y_obs,
                              const int& thinning_num,
                              const int& K,
                              const int& M,
                              const int& tot_mcmc_iters,
                              const int& r_stored_iters,
                              const int& n_temp_trans,
                              const arma::vec& c,
                              const double& b,
                              const double& nu_1,
                              const double& alpha1l,
                              const double& alpha2l,
                              const double& beta1l,
                              const double& beta2l,
                              const double& a_Z_PM,
                              const double& a_pi_PM,
                              const double& var_alpha3,
                              const double& var_epsilon1,
                              const double& var_epsilon2,
                              const double& alpha,
                              const double& beta,
                              const double& alpha_0,
                              const double& beta_0,
                              const std::string directory,
                              const double& beta_N_t,
                              const int& N_t){
  int P = y_obs.n_cols;
  int n_obs = y_obs.n_rows;

  arma::cube nu(K, P, r_stored_iters, arma::fill::randn);
  arma::cube chi(n_obs, M, r_stored_iters, arma::fill::randn);
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(r_stored_iters);
  arma::cube Z = arma::randi<arma::cube>(n_obs, K, r_stored_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_obs; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  // Create parameters for tempered transitions using geometric scheme
  arma::vec beta_ladder(N_t, arma::fill::ones);
  beta_ladder(N_t - 1) = beta_N_t;
  double geom_mult = std::pow(beta_N_t, 1.0/N_t);
  Rcpp::Rcout << "geom_mult: " << geom_mult << "\n";
  for(int i = 1; i < N_t; i++){
    beta_ladder(i) = beta_ladder(i-1) * geom_mult;
    // beta_ladder(i) = 1 - ((1- beta_N_t) *(std::pow(i/ (N_t - 1.0), 2.0)));
    Rcpp::Rcout << "beta_i: " << beta_ladder(i) << "\n";
  }
  // Create storage for tempered transitions
  arma::cube nu_TT(K, P, (2 * N_t) + 1, arma::fill::randn);
  arma::cube chi_TT(n_obs, M, (2 * N_t) + 1, arma::fill::randn);
  arma::mat pi_TT(K, (2 * N_t) + 1, arma::fill::zeros);
  arma::vec sigma_TT((2 * N_t) + 1, arma::fill::ones);
  arma::cube Z_TT = arma::randi<arma::cube>(n_obs, K, (2 * N_t) + 1,
                                            arma::distr_param(0,1));
  arma::mat delta_TT(M, (2 * N_t) + 1, arma::fill::ones);
  arma::field<arma::cube> gamma_TT((2 * N_t) + 1, 1);
  arma::field<arma::cube> Phi_TT((2 * N_t) + 1, 1);
  arma::mat A_TT = arma::ones((2 * N_t) + 1, 2);
  arma::vec alpha_3_TT = arma::ones((2 * N_t) + 1);

  for(int i = 0; i < ((2 * N_t) + 1); i++){
    gamma_TT(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi_TT(i,0) = arma::randn(K, P, M);
  }

  arma::mat tau_TT((2 * N_t) + 1, K, arma::fill::ones);

  int temp_ind = 0;
  double logA = 0;
  double logu = 0;
  int accept_num = 0;

  for(int i=0; i < tot_mcmc_iters; i++){
    if(((i % n_temp_trans) != 0) || (i == 0)){
      updateZ_PMMV(y_obs, Phi((i % r_stored_iters),0),
                   nu.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                   pi.col((i % r_stored_iters)), sigma((i % r_stored_iters)),
                   (i % r_stored_iters), r_stored_iters, alpha_3(i % r_stored_iters),
                   a_Z_PM, Z_ph, Z);

      updatePi_PM(alpha_3(i % r_stored_iters) ,Z.slice(i% r_stored_iters), c,
                  (i % r_stored_iters), r_stored_iters, a_pi_PM, pi_ph, pi);

      updateAlpha3(pi.col(i % r_stored_iters), b, Z.slice(i % r_stored_iters),
                   (i % r_stored_iters), r_stored_iters, var_alpha3, alpha_3);

      tilde_tau(0) = delta(0, (i % r_stored_iters));
      for(int j = 1; j < M; j++){
        tilde_tau(j) = tilde_tau(j-1) * delta(j,(i % r_stored_iters));
      }

      updatePhiMV(y_obs, nu.slice((i % r_stored_iters)),
                  gamma((i % r_stored_iters),0), tilde_tau,
                  Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                  sigma((i % r_stored_iters)), (i % r_stored_iters),
                  r_stored_iters, m_1, M_1, Phi);

      updateDelta(Phi((i % r_stored_iters),0), gamma((i % r_stored_iters),0),
                  A.row(i % r_stored_iters).t(), (i % r_stored_iters),
                  r_stored_iters, delta);

      updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i % r_stored_iters)),
              var_epsilon1, var_epsilon2, (i % r_stored_iters), r_stored_iters, A);

      updateGamma(nu_1, delta.col((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                  (i % r_stored_iters), r_stored_iters, gamma);

      updateNuMV(y_obs, tau.row((i % r_stored_iters)).t(),
                 Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)),
                 chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)),
                 (i % r_stored_iters), r_stored_iters, b_1, B_1, nu);

      updateTauMV(alpha, beta, nu.slice((i % r_stored_iters)), (i % r_stored_iters),
                  r_stored_iters, tau);

      updateSigmaMV(y_obs, alpha_0, beta_0,
                    nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                    Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                    (i % r_stored_iters), r_stored_iters, sigma);

      updateChiMV(y_obs, Phi((i % r_stored_iters),0),
                  nu.slice((i % r_stored_iters)), Z.slice((i % r_stored_iters)),
                  sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters,
                  chi);
    }
    if((i % n_temp_trans) == 0 && (i > 0)){
      // initialize placeholders
      nu_TT.slice(0) = nu.slice (i % r_stored_iters);
      chi_TT.slice(0) = chi.slice(i % r_stored_iters);
      pi_TT.col(0) = pi.col(i % r_stored_iters);
      sigma_TT(0) = sigma(i % r_stored_iters);
      Z_TT.slice(0) = Z.slice(i % r_stored_iters);
      delta_TT.col(0) = delta.col(i % r_stored_iters);
      gamma_TT(0,0) = gamma(i % r_stored_iters,0);
      Phi_TT(0,0) = Phi(i % r_stored_iters,0);
      A_TT.row(0) = A.row(i % r_stored_iters);
      tau_TT.row(0) = tau.row(i % r_stored_iters);
      alpha_3_TT(0) = alpha_3(i % r_stored_iters);

      nu_TT.slice(1) = nu.slice(i % r_stored_iters);
      chi_TT.slice(1) = chi.slice(i % r_stored_iters);
      pi_TT.col(1) = pi.col(i % r_stored_iters);
      sigma_TT(1) = sigma(i % r_stored_iters);
      Z_TT.slice(1) = Z.slice(i % r_stored_iters);
      delta_TT.col(1) = delta.col(i % r_stored_iters);
      gamma_TT(1,0) = gamma(i % r_stored_iters,0);
      Phi_TT(1,0) = Phi(i % r_stored_iters,0);
      A_TT.row(1) = A.row(i % r_stored_iters);
      tau_TT.row(1) = tau.row(i % r_stored_iters);
      alpha_3_TT(1) = alpha_3(i % r_stored_iters);

      temp_ind = 0;

      // Perform tempered transitions
      for(int l = 1; l < ((2 * N_t) + 1); l++){
        updateZTempered_PMMV(beta_ladder(temp_ind), y_obs,
                             Phi_TT(l,0), nu_TT.slice(l), chi_TT.slice(l),
                             pi_TT.col(l), sigma_TT(l), l, (2 * N_t) + 1, alpha_3_TT(l), a_Z_PM,
                             Z_ph, Z_TT);
        updatePi_PM(alpha_3_TT(l), Z_TT.slice(l), c, l, (2 * N_t) + 1, a_pi_PM, pi_ph, pi_TT);
        updateAlpha3(pi_TT.col(l), b, Z_TT.slice(l), l, (2 * N_t) + 1, var_alpha3, alpha_3_TT);

        tilde_tau(0) = delta_TT(0, l);
        for(int j = 1; j < M; j++){
          tilde_tau(j) = tilde_tau(j-1) * delta_TT(j,l);
        }

        updatePhiTemperedMV(beta_ladder(temp_ind), y_obs,
                            nu_TT.slice(l), gamma_TT(l,0), tilde_tau, Z_TT.slice(l),
                            chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, m_1, M_1,
                            Phi_TT);
        updateDelta(Phi_TT(l,0), gamma_TT(l,0), A_TT.row(l).t(), l, (2 * N_t) + 1,
                    delta_TT);

        updateA(alpha1l, beta1l, alpha2l, beta2l, delta_TT.col(l), var_epsilon1,
                var_epsilon2, l, (2 * N_t) + 1, A_TT);
        updateGamma(nu_1, delta_TT.col(l), Phi_TT(l,0), l, (2 * N_t) + 1,
                    gamma_TT);
        updateNuTemperedMV(beta_ladder(temp_ind), y_obs,
                           tau_TT.row(l).t(), Phi_TT(l,0), Z_TT.slice(l),
                           chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1,
                           b_1, B_1, nu_TT);
        updateTauMV(alpha, beta, nu_TT.slice(l), l, (2 * N_t) + 1, tau_TT);
        updateSigmaTemperedMV(beta_ladder(temp_ind), y_obs,
                              alpha_0, beta_0, nu_TT.slice(l), Phi_TT(l,0),
                              Z_TT.slice(l), chi_TT.slice(l), l, (2 * N_t) + 1,
                              sigma_TT);
        updateChiTemperedMV(beta_ladder(temp_ind), y_obs,
                            Phi_TT(l,0), nu_TT.slice(l), Z_TT.slice(l), sigma_TT(l),
                            l, (2 * N_t) + 1, chi_TT);

        // update temp_ind
        if(l < N_t){
          temp_ind = temp_ind + 1;
        }
        if(l > N_t){
          temp_ind = temp_ind - 1;
        }
      }
      logA = CalculateTTAcceptanceMV(beta_ladder, y_obs,
                                     nu_TT, Phi_TT, Z_TT, chi_TT, sigma_TT);
      logu = std::log(R::runif(0,1));

      Rcpp::Rcout << "prob_accept: " << logA<< "\n";
      Rcpp::Rcout << "logu: " << logu<< "\n";

      if(logu < logA){
        Rcpp::Rcout << "Accept \n";
        nu.slice(i % r_stored_iters) = nu_TT.slice(2 * N_t);
        chi.slice(i % r_stored_iters) = chi_TT.slice(2 * N_t);
        pi.col(i % r_stored_iters) = pi_TT.col(2 * N_t);
        sigma(i % r_stored_iters) = sigma_TT(2 * N_t);
        Z.slice(i % r_stored_iters) = Z_TT.slice(2 * N_t);
        delta.col(i % r_stored_iters) = delta_TT.col(2 * N_t);
        gamma(i % r_stored_iters,0) = gamma_TT(2 * N_t,0);
        Phi(i % r_stored_iters,0) = Phi_TT(2 * N_t,0);
        A.row(i % r_stored_iters) = A_TT.row(2 * N_t);
        tau.row(i % r_stored_iters) = tau_TT.row(2 * N_t);
        alpha_3(i % r_stored_iters) = alpha_3_TT(2 * N_t);

        //update accept number
        accept_num = accept_num + 1;
      }

      //initialize next state
      if(((i+1) % r_stored_iters) != 0){
        nu.slice((i+1) % r_stored_iters) = nu.slice(i % r_stored_iters);
        chi.slice((i+1) % r_stored_iters) = chi.slice(i % r_stored_iters);
        pi.col((i+1) % r_stored_iters) = pi.col(i % r_stored_iters);
        sigma((i+1) % r_stored_iters) = sigma(i % r_stored_iters);
        Z.slice((i+1) % r_stored_iters) = Z.slice(i % r_stored_iters);
        delta.col((i+1) % r_stored_iters) = delta.col(i % r_stored_iters);
        A.row((i+1) % r_stored_iters) = A.row(i % r_stored_iters);
        tau.row((i+1) % r_stored_iters) = tau.row(i % r_stored_iters);
        Phi((i+1) % r_stored_iters,0) = Phi(i % r_stored_iters, 0);
        alpha_3((i+1) % r_stored_iters) =  alpha_3(i % r_stored_iters);
      }
    }
    loglik((i % r_stored_iters)) =  calcLikelihoodMV(y_obs,
           nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
           Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
           sigma((i % r_stored_iters)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Accpetance Probability: " << accept_num / (std::round(i / n_temp_trans)) << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i % r_stored_iters)-4, (i % r_stored_iters))) << "\n";
      Rcpp::checkUserInterrupt();
    }
    if(((i+1) % r_stored_iters) == 0 && i > 1){
      // Save parameters
      arma::cube nu1(K, P, r_stored_iters/thinning_num, arma::fill::randn);
      arma::cube chi1(n_obs, M, r_stored_iters/thinning_num, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec alpha_31(r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters/thinning_num, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters/thinning_num, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_obs, K, r_stored_iters/thinning_num,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters/thinning_num, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters/thinning_num,1);
      arma::field<arma::cube> Phi1(r_stored_iters/thinning_num, 1);
      arma::mat tau1(r_stored_iters/thinning_num, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);

      for(int p=1; p < r_stored_iters / thinning_num; p++){
        nu1.slice(p) = nu.slice(thinning_num*p - 1);
        chi1.slice(p) = chi.slice(thinning_num*p - 1);
        pi1.col(p) = pi.col(thinning_num*p - 1);
        alpha_31(p) = alpha_3(thinning_num*p - 1);
        sigma1(p) = sigma(thinning_num*p - 1);
        A1.row(p) = A.row(thinning_num*p - 1);
        Z1.slice(p) = Z.slice(thinning_num*p - 1);
        delta1.col(p) = delta.col(thinning_num*p - 1);
        gamma1(p,0) = gamma(thinning_num*p - 1,0);
        Phi1(p,0) = Phi(thinning_num*p  - 1,0);
        tau1.row(p) = tau.row(thinning_num*p - 1);

        nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
        chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
        pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
        alpha_31.save(directory + "alpha_3" + std::to_string(q) + ".txt", arma::arma_ascii);
        A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
        delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
        sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
        tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
        gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
        Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
        Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

        //reset all parameters
        nu.slice(0) = nu.slice(i % r_stored_iters);
        chi.slice(0) = chi.slice(i % r_stored_iters);
        pi.col(0) = pi.col(i % r_stored_iters);
        alpha_3(0) = alpha_3(i % r_stored_iters);
        A.row(0) = A.row(i % r_stored_iters);
        delta.col(0) = delta.col(i % r_stored_iters);
        sigma(0) = sigma(i % r_stored_iters);
        tau.row(0) = tau.row(i % r_stored_iters);
        gamma(0,0) = gamma(i % r_stored_iters, 0);
        Phi(0,0) = Phi(i % r_stored_iters, 0);
        Z.slice(0) = Z.slice(i % r_stored_iters);

        q = q + 1;
      }
    }
  }

  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu_TT),
                                         Rcpp::Named("alpha_3", alpha_3_TT),
                                         Rcpp::Named("chi", chi_TT),
                                         Rcpp::Named("pi", pi_TT),
                                         Rcpp::Named("A", A_TT),
                                         Rcpp::Named("delta", delta_TT),
                                         Rcpp::Named("sigma", sigma_TT),
                                         Rcpp::Named("tau", tau_TT),
                                         Rcpp::Named("gamma", gamma_TT),
                                         Rcpp::Named("Phi", Phi_TT),
                                         Rcpp::Named("Z", Z_TT),
                                         Rcpp::Named("loglik", loglik));
  return params;
}

// Conducts un-tempered MCMC to mean and allocation parameters for the multivariate model
//
// @name BFPMM_Nu_ZMV
// @param y_obs Matrix of observed vectors
// @param K Int containing the number of clusters
// @param M Int containing the number of eigenvectors
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int containing number of iterations performed for each batch
// @param c Vector containing hyperparmeters for pi
// @param b double containing hyperparameter for alpha_3
// @param a_12 Vector containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param var_pi Double containing variance parameter of the random walk MH for pi parameter
// @param var_Z Double containing variance parameter of the random walk MH for Z parameter
// @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param directory String containing path to store batches of MCMC samples
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_Nu_ZMV(const arma::mat& y_obs,
                               const int& K,
                               const int& M,
                               const int& tot_mcmc_iters,
                               const arma::vec& c,
                               const double& b,
                               const double& alpha1l,
                               const double& alpha2l,
                               const double& beta1l,
                               const double& beta2l,
                               const double& a_Z_PM,
                               const double& a_pi_PM,
                               const double& var_alpha3,
                               const double& var_epsilon1,
                               const double& var_epsilon2,
                               const double& alpha,
                               const double& beta,
                               const double& alpha_0,
                               const double& beta_0){
  int P = y_obs.n_cols;
  int n_obs = y_obs.n_rows;

  arma::cube nu(K, P, tot_mcmc_iters, arma::fill::randn);
  arma::cube chi(n_obs, M, tot_mcmc_iters, arma::fill::zeros);
  arma::mat pi(K, tot_mcmc_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(tot_mcmc_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(tot_mcmc_iters);
  arma::cube Z = arma::randi<arma::cube>(n_obs, K, tot_mcmc_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_obs; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, tot_mcmc_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(tot_mcmc_iters,1);
  arma::field<arma::cube> Phi(tot_mcmc_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(tot_mcmc_iters, 2);
  arma::vec loglik = arma::zeros(tot_mcmc_iters);

  for(int i = 0; i < tot_mcmc_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::zeros(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(tot_mcmc_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  for(int i = 0; i < tot_mcmc_iters; i++){
    updateZ_PMMV(y_obs, Phi(i,0),
                 nu.slice(i), chi.slice(i),
                 pi.col(i), sigma(i),
                 i, tot_mcmc_iters, alpha_3(i),
                 a_Z_PM, Z_ph, Z);
    updatePi_PM(alpha_3(i) ,Z.slice(i), c,
                (i), tot_mcmc_iters, a_pi_PM, pi_ph, pi);

    updateAlpha3(pi.col(i), b, Z.slice(i),
                 (i), tot_mcmc_iters, var_alpha3, alpha_3);

    tilde_tau(0) = delta(0, (i));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i));
    }

    updateNuMV(y_obs, tau.row((i)).t(),
               Phi((i),0), Z.slice((i)),
               chi.slice((i)), sigma((i)),
               (i), tot_mcmc_iters, b_1, B_1, nu);

    updateTauMV(alpha, beta, nu.slice((i)), (i),
                tot_mcmc_iters, tau);

    updateSigmaMV(y_obs, alpha_0, beta_0,
                  nu.slice((i)), Phi((i),0),
                  Z.slice((i)), chi.slice((i)),
                  (i), tot_mcmc_iters, sigma);

    // Calculate log likelihood
    loglik((i)) =  calcLikelihoodMV(y_obs, nu.slice((i)),
           Phi((i),0), Z.slice((i)), chi.slice((i)), sigma((i)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i)-19, (i))) << "\n";
      Rcpp::checkUserInterrupt();
    }
  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
                                         Rcpp::Named("pi", pi),
                                         Rcpp::Named("alpha_3", alpha_3),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("Z", Z),
                                         Rcpp::Named("loglik", loglik));
  return params;
}

// Conducts un-tempered MCMC to estimate the posterior distribution of parameters not related to Z or Nu, conditioned on a value of Nu and Z
//
// @name BFPMM_Theta
// @param y_obs Matrix of observed vectors
// @param K Int containing the number of clusters
// @param basis degree Int containing the degree of B-splines used
// @param M Int containing the number of eigenfunctions
// @param boundary_knots Vector containing the boundary points of our index domain of interest
// @param internal_knots Vector location of internal knots for B-splines
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param c Vector containing hyperparmeters for pi
// @param b double containing hyperparameter for alpha_3
// @param a_12 Vector containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param var_pi Double containing variance parameter of the random walk MH for pi parameter
// @param var_Z Double containing variance parameter of the random walk MH for Z parameter
// @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param Z_est Matrix containing Z values to be conditioned on
// @param nu_est Matrix containing nu values to be conditioned on
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_ThetaMV(const arma::mat& y_obs,
                                const int& K,
                                const int& M,
                                const int& tot_mcmc_iters,
                                const arma::vec& c,
                                const double& b,
                                const double& nu_1,
                                const double& alpha1l,
                                const double& alpha2l,
                                const double& beta1l,
                                const double& beta2l,
                                const double& a_Z_PM,
                                const double& a_pi_PM,
                                const double& var_alpha3,
                                const double& var_epsilon1,
                                const double& var_epsilon2,
                                const double& alpha,
                                const double& beta,
                                const double& alpha_0,
                                const double& beta_0,
                                const arma::mat& Z_est,
                                const arma::mat& nu_est){

  int n_obs = y_obs.n_rows;
  int P = y_obs.n_cols;

  arma::cube nu(K, P, tot_mcmc_iters, arma::fill::randn);
  arma::cube chi(n_obs, M, tot_mcmc_iters, arma::fill::randn);
  arma::mat pi(K, tot_mcmc_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(tot_mcmc_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(tot_mcmc_iters);
  arma::cube Z = arma::randi<arma::cube>(n_obs, K, tot_mcmc_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_obs; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, tot_mcmc_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(tot_mcmc_iters,1);
  arma::field<arma::cube> Phi(tot_mcmc_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(tot_mcmc_iters, 2);
  arma::vec loglik = arma::zeros(tot_mcmc_iters);

  for(int i = 0; i < tot_mcmc_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(tot_mcmc_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  Z.slice(0) = Z_est;
  nu.slice(0) = nu_est;

  for(int i = 1; i < tot_mcmc_iters; i++){
    Z.slice(i) = Z_est;
    nu.slice(i) = nu_est;
  }


  for(int i = 0; i < tot_mcmc_iters; i++){
    tilde_tau(0) = delta(0, (i));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i));
    }

    updatePhiMV(y_obs, nu.slice((i)),
                gamma((i),0), tilde_tau,
                Z.slice((i)), chi.slice((i)),
                sigma((i)), (i),
                tot_mcmc_iters, m_1, M_1, Phi);

    updateDelta(Phi((i),0), gamma((i),0),
                A.row(i).t(), (i),
                tot_mcmc_iters, delta);

    updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i)),
            var_epsilon1, var_epsilon2, (i), tot_mcmc_iters, A);

    updateGamma(nu_1, delta.col((i)), Phi((i),0),
                (i), tot_mcmc_iters, gamma);

    updateTauMV(alpha, beta, nu.slice((i)), (i),
                tot_mcmc_iters, tau);

    updateSigmaMV(y_obs, alpha_0, beta_0,
                  nu.slice((i)), Phi((i),0),
                  Z.slice((i)), chi.slice((i)),
                  (i), tot_mcmc_iters, sigma);

    updateChiMV(y_obs, Phi((i),0),
                nu.slice((i)), Z.slice((i)),
                sigma((i)), (i), tot_mcmc_iters,
                chi);

    // Calculate log likelihood
    loglik((i)) =  calcLikelihoodMV(y_obs, nu.slice((i)),
           Phi((i),0), Z.slice((i)), chi.slice((i)), sigma((i)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i)-99, (i))) << "\n";
      Rcpp::checkUserInterrupt();
    }
  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("Z", Z),
                                         Rcpp::Named("nu", nu),
                                         Rcpp::Named("chi", chi),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("gamma", gamma),
                                         Rcpp::Named("Phi", Phi),
                                         Rcpp::Named("loglik", loglik));
  return params;
}

// Conducts a mixture of untempered sampling and termpered sampling to get posterior draws from the partial membership model
//
// @name BFPMM_MTT_warm_startMV
// @param y_obs Matrix of observed vectors
// @param thinning_num Int containing how often we save an MCMC iteration
// @param K Int containing the number of clusters
// @param M Int containing the number of eigenfunctions
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int constaining number of iterations performed for each batch
// @param t_star Field (list) of vectors containing time points of interest that are not observed (optional)
// @param rho Double containing hyperparmater for sampling from Z
// @param alpha_3 Double hyperparameter for sampling from pi
// @param a_12 Vec containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param a_Z_PM Double containing hyperparameter used to sample from the posterior of Z
// @param a_pi_PM Double containing hyperparameter used to sample from the posterior of pi
// @param var_alpha3 Doubel containing hyperparameter for sampling from alpha_3
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param directory String containing path to store batches of MCMC samples
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BFPMM_MTT_warm_startMV(const arma::mat& y_obs,
                                         const int& thinning_num,
                                         const int& K,
                                         const int& M,
                                         const int& tot_mcmc_iters,
                                         const int& r_stored_iters,
                                         const int& n_temp_trans,
                                         const arma::vec& c,
                                         const double& b,
                                         const double& nu_1,
                                         const double& alpha1l,
                                         const double& alpha2l,
                                         const double& beta1l,
                                         const double& beta2l,
                                         const double& a_Z_PM,
                                         const double& a_pi_PM,
                                         const double& var_alpha3,
                                         const double& var_epsilon1,
                                         const double& var_epsilon2,
                                         const double& alpha,
                                         const double& beta,
                                         const double& alpha_0,
                                         const double& beta_0,
                                         const std::string directory,
                                         const double& beta_N_t,
                                         const int& N_t,
                                         const arma::mat& Z_est,
                                         const arma::vec& pi_est,
                                         const double& alpha_3_est,
                                         const arma::vec& delta_est,
                                         const arma::cube& gamma_est,
                                         const arma::cube& Phi_est,
                                         const arma::vec& A_est,
                                         const arma::mat& nu_est,
                                         const arma::vec& tau_est,
                                         const double& sigma_est,
                                         const arma::mat& chi_est){
  int n_obs = y_obs.n_rows;
  int P = y_obs.n_cols;

  arma::cube nu(K, P, r_stored_iters, arma::fill::randn);
  arma::cube chi(n_obs, M, r_stored_iters, arma::fill::randn);
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(r_stored_iters);
  arma::cube Z = arma::randi<arma::cube>(n_obs, K, r_stored_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_obs; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  // Create parameters for tempered transitions using geometric scheme
  arma::vec beta_ladder(N_t, arma::fill::ones);
  beta_ladder(N_t - 1) = beta_N_t;
  double geom_mult = std::pow(beta_N_t, 1.0/N_t);
  Rcpp::Rcout << "geom_mult: " << geom_mult << "\n";
  for(int i = 1; i < N_t; i++){
    beta_ladder(i) = beta_ladder(i-1) * geom_mult;
    // beta_ladder(i) = 1 - ((1- beta_N_t) *(std::pow(i/ (N_t - 1.0), 2.0)));
    Rcpp::Rcout << "beta_i: " << beta_ladder(i) << "\n";
  }
  // Create storage for tempered transitions
  arma::cube nu_TT(K, P, (2 * N_t) + 1, arma::fill::randn);
  arma::cube chi_TT(n_obs, M, (2 * N_t) + 1, arma::fill::randn);
  arma::mat pi_TT(K, (2 * N_t) + 1, arma::fill::zeros);
  arma::vec sigma_TT((2 * N_t) + 1, arma::fill::ones);
  arma::cube Z_TT = arma::randi<arma::cube>(n_obs, K, (2 * N_t) + 1,
                                            arma::distr_param(0,1));
  arma::mat delta_TT(M, (2 * N_t) + 1, arma::fill::ones);
  arma::field<arma::cube> gamma_TT((2 * N_t) + 1, 1);
  arma::field<arma::cube> Phi_TT((2 * N_t) + 1, 1);
  arma::mat A_TT = arma::ones((2 * N_t) + 1, 2);
  arma::vec alpha_3_TT = arma::ones((2 * N_t) + 1);

  for(int i = 0; i < ((2 * N_t) + 1); i++){
    gamma_TT(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi_TT(i,0) = arma::randn(K, P, M);
  }

  arma::mat tau_TT((2 * N_t) + 1, K, arma::fill::ones);

  int temp_ind = 0;
  double logA = 0;
  double logu = 0;
  int accept_num = 0;

  Z.slice(0) = Z_est;;
  pi.col(0) = pi_est;

  alpha_3(0) = alpha_3_est;
  delta.col(0) = delta_est;
  gamma(0,0) = gamma_est;
  Phi(0,0) = Phi_est;
  A.row(0) = A_est.t();
  nu.slice(0) = nu_est;
  tau.row(0) = tau_est.t();
  sigma(0) = sigma_est;

  chi.slice(0) = chi_est;

  for(int i=0; i < tot_mcmc_iters; i++){
    if(((i % n_temp_trans) != 0) || (i == 0)){
      updateZ_PMMV(y_obs, Phi((i % r_stored_iters),0),
                   nu.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                   pi.col((i % r_stored_iters)), sigma((i % r_stored_iters)),
                   (i % r_stored_iters), r_stored_iters, alpha_3(i % r_stored_iters),
                   a_Z_PM, Z_ph, Z);

      updatePi_PM(alpha_3(i % r_stored_iters) ,Z.slice(i% r_stored_iters), c,
                  (i % r_stored_iters), r_stored_iters, a_pi_PM, pi_ph, pi);

      updateAlpha3(pi.col(i % r_stored_iters), b, Z.slice(i % r_stored_iters),
                   (i % r_stored_iters), r_stored_iters, var_alpha3, alpha_3);

      tilde_tau(0) = delta(0, (i % r_stored_iters));
      for(int j = 1; j < M; j++){
        tilde_tau(j) = tilde_tau(j-1) * delta(j,(i % r_stored_iters));
      }

      updatePhiMV(y_obs, nu.slice((i % r_stored_iters)),
                  gamma((i % r_stored_iters),0), tilde_tau,
                  Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                  sigma((i % r_stored_iters)), (i % r_stored_iters),
                  r_stored_iters, m_1, M_1, Phi);

      updateDelta(Phi((i % r_stored_iters),0), gamma((i % r_stored_iters),0),
                  A.row(i % r_stored_iters).t(), (i % r_stored_iters),
                  r_stored_iters, delta);

      updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i % r_stored_iters)),
              var_epsilon1, var_epsilon2, (i % r_stored_iters), r_stored_iters, A);

      updateGamma(nu_1, delta.col((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                  (i % r_stored_iters), r_stored_iters, gamma);

      updateNuMV(y_obs, tau.row((i % r_stored_iters)).t(),
                 Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)),
                 chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)),
                 (i % r_stored_iters), r_stored_iters, b_1, B_1, nu);

      updateTauMV(alpha, beta, nu.slice((i % r_stored_iters)), (i % r_stored_iters),
                  r_stored_iters, tau);

      updateSigmaMV(y_obs, alpha_0, beta_0,
                    nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
                    Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                    (i % r_stored_iters), r_stored_iters, sigma);

      updateChiMV(y_obs, Phi((i % r_stored_iters),0),
                  nu.slice((i % r_stored_iters)), Z.slice((i % r_stored_iters)),
                  sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters,
                  chi);
    }

    if((i % n_temp_trans) == 0 && (i > 0)){
      // initialize placeholders
      nu_TT.slice(0) = nu.slice (i % r_stored_iters);
      chi_TT.slice(0) = chi.slice(i % r_stored_iters);
      pi_TT.col(0) = pi.col(i % r_stored_iters);
      sigma_TT(0) = sigma(i % r_stored_iters);
      Z_TT.slice(0) = Z.slice(i % r_stored_iters);
      delta_TT.col(0) = delta.col(i % r_stored_iters);
      gamma_TT(0,0) = gamma(i % r_stored_iters,0);
      Phi_TT(0,0) = Phi(i % r_stored_iters,0);
      A_TT.row(0) = A.row(i % r_stored_iters);
      tau_TT.row(0) = tau.row(i % r_stored_iters);
      alpha_3_TT(0) = alpha_3(i % r_stored_iters);

      nu_TT.slice(1) = nu.slice(i % r_stored_iters);
      chi_TT.slice(1) = chi.slice(i % r_stored_iters);
      pi_TT.col(1) = pi.col(i % r_stored_iters);
      sigma_TT(1) = sigma(i % r_stored_iters);
      Z_TT.slice(1) = Z.slice(i % r_stored_iters);
      delta_TT.col(1) = delta.col(i % r_stored_iters);
      gamma_TT(1,0) = gamma(i % r_stored_iters,0);
      Phi_TT(1,0) = Phi(i % r_stored_iters,0);
      A_TT.row(1) = A.row(i % r_stored_iters);
      tau_TT.row(1) = tau.row(i % r_stored_iters);
      alpha_3_TT(1) = alpha_3(i % r_stored_iters);

      temp_ind = 0;

      // Perform tempered transitions
      for(int l = 1; l < ((2 * N_t) + 1); l++){
        updateZTempered_PMMV(beta_ladder(temp_ind), y_obs,
                             Phi_TT(l,0), nu_TT.slice(l), chi_TT.slice(l),
                             pi_TT.col(l), sigma_TT(l), l, (2 * N_t) + 1, alpha_3_TT(l), a_Z_PM,
                             Z_ph, Z_TT);
        updatePi_PM(alpha_3_TT(l), Z_TT.slice(l), c, l, (2 * N_t) + 1, a_pi_PM, pi_ph, pi_TT);
        updateAlpha3(pi_TT.col(l), b, Z_TT.slice(l), l, (2 * N_t) + 1, var_alpha3, alpha_3_TT);

        tilde_tau(0) = delta_TT(0, l);
        for(int j = 1; j < M; j++){
          tilde_tau(j) = tilde_tau(j-1) * delta_TT(j,l);
        }

        updatePhiTemperedMV(beta_ladder(temp_ind), y_obs,
                            nu_TT.slice(l), gamma_TT(l,0), tilde_tau, Z_TT.slice(l),
                            chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, m_1, M_1,
                            Phi_TT);
        updateDelta(Phi_TT(l,0), gamma_TT(l,0), A_TT.row(l).t(), l, (2 * N_t) + 1,
                    delta_TT);

        updateA(alpha1l, beta1l, alpha2l, beta2l, delta_TT.col(l), var_epsilon1,
                var_epsilon2, l, (2 * N_t) + 1, A_TT);
        updateGamma(nu_1, delta_TT.col(l), Phi_TT(l,0), l, (2 * N_t) + 1,
                    gamma_TT);
        updateNuTemperedMV(beta_ladder(temp_ind), y_obs,
                           tau_TT.row(l).t(), Phi_TT(l,0), Z_TT.slice(l),
                           chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1,
                           b_1, B_1, nu_TT);
        updateTauMV(alpha, beta, nu_TT.slice(l), l, (2 * N_t) + 1, tau_TT);
        updateSigmaTemperedMV(beta_ladder(temp_ind), y_obs,
                              alpha_0, beta_0, nu_TT.slice(l), Phi_TT(l,0),
                              Z_TT.slice(l), chi_TT.slice(l), l, (2 * N_t) + 1,
                              sigma_TT);
        updateChiTemperedMV(beta_ladder(temp_ind), y_obs,
                            Phi_TT(l,0), nu_TT.slice(l), Z_TT.slice(l), sigma_TT(l),
                            l, (2 * N_t) + 1, chi_TT);
        // update temp_ind
        if(l < N_t){
          temp_ind = temp_ind + 1;
        }
        if(l > N_t){
          temp_ind = temp_ind - 1;
        }
      }
      logA = CalculateTTAcceptanceMV(beta_ladder, y_obs,
                                     nu_TT, Phi_TT, Z_TT, chi_TT, sigma_TT);
      logu = std::log(R::runif(0,1));

      Rcpp::Rcout << "prob_accept: " << logA<< "\n";
      Rcpp::Rcout << "logu: " << logu<< "\n";

      if(logu < logA){
        Rcpp::Rcout << "Accept \n";
        nu.slice(i % r_stored_iters) = nu_TT.slice(2 * N_t);
        chi.slice(i % r_stored_iters) = chi_TT.slice(2 * N_t);
        pi.col(i % r_stored_iters) = pi_TT.col(2 * N_t);
        sigma(i % r_stored_iters) = sigma_TT(2 * N_t);
        Z.slice(i % r_stored_iters) = Z_TT.slice(2 * N_t);
        delta.col(i % r_stored_iters) = delta_TT.col(2 * N_t);
        gamma(i % r_stored_iters,0) = gamma_TT(2 * N_t,0);
        Phi(i % r_stored_iters,0) = Phi_TT(2 * N_t,0);
        A.row(i % r_stored_iters) = A_TT.row(2 * N_t);
        tau.row(i % r_stored_iters) = tau_TT.row(2 * N_t);
        alpha_3(i % r_stored_iters) = alpha_3_TT(2 * N_t);

        //update accept number
        accept_num = accept_num + 1;
      }

      //initialize next state
      if(((i+1) % r_stored_iters) != 0){
        nu.slice((i+1) % r_stored_iters) = nu.slice(i % r_stored_iters);
        chi.slice((i+1) % r_stored_iters) = chi.slice(i % r_stored_iters);
        pi.col((i+1) % r_stored_iters) = pi.col(i % r_stored_iters);
        sigma((i+1) % r_stored_iters) = sigma(i % r_stored_iters);
        Z.slice((i+1) % r_stored_iters) = Z.slice(i % r_stored_iters);
        delta.col((i+1) % r_stored_iters) = delta.col(i % r_stored_iters);
        A.row((i+1) % r_stored_iters) = A.row(i % r_stored_iters);
        tau.row((i+1) % r_stored_iters) = tau.row(i % r_stored_iters);
        Phi((i+1) % r_stored_iters,0) = Phi(i % r_stored_iters, 0);
        alpha_3((i+1) % r_stored_iters) =  alpha_3(i % r_stored_iters);
      }
    }
    loglik((i % r_stored_iters)) =  calcLikelihoodMV(y_obs,
           nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0),
           Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
           sigma((i % r_stored_iters)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Accpetance Probability: " << accept_num / (std::round(i / n_temp_trans)) << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i % r_stored_iters)-4, (i % r_stored_iters))) << "\n";
      Rcpp::checkUserInterrupt();
    }
    if(((i+1) % r_stored_iters) == 0 && i > 1){
      // Save parameters
      arma::cube nu1(K, P, r_stored_iters/thinning_num, arma::fill::randn);
      arma::cube chi1(n_obs, M, r_stored_iters/thinning_num, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec alpha_31(r_stored_iters/thinning_num, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters/thinning_num, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters/thinning_num, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_obs, K, r_stored_iters/thinning_num,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters/thinning_num, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters/thinning_num,1);
      arma::field<arma::cube> Phi1(r_stored_iters/thinning_num, 1);
      arma::mat tau1(r_stored_iters/thinning_num, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);

      for(int p=1; p < r_stored_iters / thinning_num; p++){
        nu1.slice(p) = nu.slice(thinning_num*p - 1);
        chi1.slice(p) = chi.slice(thinning_num*p - 1);
        pi1.col(p) = pi.col(thinning_num*p - 1);
        alpha_31(p) = alpha_3(thinning_num*p - 1);
        sigma1(p) = sigma(thinning_num*p - 1);
        A1.row(p) = A.row(thinning_num*p - 1);
        Z1.slice(p) = Z.slice(thinning_num*p - 1);
        delta1.col(p) = delta.col(thinning_num*p - 1);
        gamma1(p,0) = gamma(thinning_num*p - 1,0);
        Phi1(p,0) = Phi(thinning_num*p  - 1,0);
        tau1.row(p) = tau.row(thinning_num*p - 1);
      }

      nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
      chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
      pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
      alpha_31.save(directory + "alpha_3" + std::to_string(q) + ".txt", arma::arma_ascii);
      A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
      delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
      sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
      tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
      gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
      Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
      Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

      //reset all parameters
      nu.slice(0) = nu.slice(i % r_stored_iters);
      chi.slice(0) = chi.slice(i % r_stored_iters);
      pi.col(0) = pi.col(i % r_stored_iters);
      alpha_3(0) = alpha_3(i % r_stored_iters);
      A.row(0) = A.row(i % r_stored_iters);
      delta.col(0) = delta.col(i % r_stored_iters);
      sigma(0) = sigma(i % r_stored_iters);
      tau.row(0) = tau.row(i % r_stored_iters);
      gamma(0,0) = gamma(i % r_stored_iters, 0);
      Phi(0,0) = Phi(i % r_stored_iters, 0);
      Z.slice(0) = Z.slice(i % r_stored_iters);

      q = q + 1;
    }
  }

  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
                                         Rcpp::Named("alpha_3", alpha_3),
                                         Rcpp::Named("chi", chi),
                                         Rcpp::Named("pi", pi),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("gamma", gamma),
                                         Rcpp::Named("Phi", Phi),
                                         Rcpp::Named("Z", Z),
                                         Rcpp::Named("loglik", loglik));
  return params;
}


// Conducts un-tempered MCMC to mean and allocation parameters for multivariate functional data
//
// @name BMFPMM_Nu_Z
// @param y_obs Field (list) of vectors containing the observed values
// @param t_obs field of matrices that contain the observed time points (each column is a dimension)
// @param n_funct Int containing number of functions observed
// @param K Int containing the number of clusters
// @param basis_degree vector containing the desired basis degree for each dimension
// @param M Int containing the number of eigenfunctions
// @param boundary_knots matrix containing the boundary knots for each dimension (each row is a dimension)
// @param internal_knots field of vectors containing the internal knots for each dimension
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param r_stored_iters Int containing number of iterations performed for each batch
// @param c Vector containing hyperparmeters for pi
// @param b double containing hyperparameter for alpha_3
// @param a_12 Vector containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param var_pi Double containing variance parameter of the random walk MH for pi parameter
// @param var_Z Double containing variance parameter of the random walk MH for Z parameter
// @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param directory String containing path to store batches of MCMC samples
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BMFPMM_Nu_Z(const arma::field<arma::vec>& y_obs,
                              const arma::field<arma::mat>& t_obs,
                              const int& n_funct,
                              const int& K,
                              const arma::vec& basis_degree,
                              const int& M,
                              const arma::mat& boundary_knots,
                              const arma::field<arma::vec>& internal_knots,
                              const int& tot_mcmc_iters,
                              const arma::vec& c,
                              const double& b,
                              const double& nu_1,
                              const double& alpha1l,
                              const double& alpha2l,
                              const double& beta1l,
                              const double& beta2l,
                              const double& a_Z_PM,
                              const double& a_pi_PM,
                              const double& var_alpha3,
                              const double& var_epsilon1,
                              const double& var_epsilon2,
                              const double& alpha,
                              const double& beta,
                              const double& alpha_0,
                              const double& beta_0){
  // Make B_obs
  arma::field<arma::mat> B_obs = TensorBSpline(t_obs, n_funct, basis_degree,
                                               boundary_knots, internal_knots);

  arma::mat P_mat = GetP(basis_degree,internal_knots);

  int P = P_mat.n_cols;
  arma::cube nu(K, P, tot_mcmc_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, tot_mcmc_iters, arma::fill::zeros);
  arma::mat pi(K, tot_mcmc_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(tot_mcmc_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(tot_mcmc_iters);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, tot_mcmc_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_funct; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, tot_mcmc_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(tot_mcmc_iters,1);
  arma::field<arma::cube> Phi(tot_mcmc_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(tot_mcmc_iters, 2);
  arma::vec loglik = arma::zeros(tot_mcmc_iters);

  for(int i = 0; i < tot_mcmc_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::zeros(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(tot_mcmc_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  for(int i = 0; i < tot_mcmc_iters; i++){
    updateZ_PM(y_obs, B_obs, Phi(i,0),
               nu.slice(i), chi.slice(i),
               pi.col(i), sigma(i),
               i, tot_mcmc_iters, alpha_3(i),
               a_Z_PM, Z_ph, Z);
    updatePi_PM(alpha_3(i) ,Z.slice(i), c,
                (i), tot_mcmc_iters, a_pi_PM, pi_ph, pi);

    updateAlpha3(pi.col(i), b, Z.slice(i),
                 (i), tot_mcmc_iters, var_alpha3, alpha_3);

    tilde_tau(0) = delta(0, (i));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i));
    }

    updateNu(y_obs, B_obs, tau.row((i)).t(),
             Phi((i),0), Z.slice((i)),
             chi.slice((i)), sigma((i)),
             (i), tot_mcmc_iters, P_mat, b_1, B_1, nu);

    updateTau(alpha, beta, nu.slice((i)), (i),
              tot_mcmc_iters, P_mat, tau);

    updateSigma(y_obs, B_obs, alpha_0, beta_0,
                nu.slice((i)), Phi((i),0),
                Z.slice((i)), chi.slice((i)),
                (i), tot_mcmc_iters, sigma);

    // Calculate log likelihood
    loglik((i)) =  calcLikelihood(y_obs, B_obs, nu.slice((i)),
           Phi((i),0), Z.slice((i)), chi.slice((i)), sigma((i)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i)-19, (i))) << "\n";
      Rcpp::checkUserInterrupt();
    }
  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
                                         Rcpp::Named("pi", pi),
                                         Rcpp::Named("alpha_3", alpha_3),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("Z", Z),
                                         Rcpp::Named("loglik", loglik));
  return params;
}

// Conducts un-tempered MCMC to estimate the posterior distribution of parameters not related to Z or Nu for multivariate functional data, conditioned on a value of Nu and Z
//
// @name BMFPMM_Theta
// @param y_obs Field (list) of vectors containing the observed values
// @param t_obs field of matrices that contain the observed time points (each column is a dimension)
// @param n_funct Int containing number of functions observed
// @param K Int containing the number of clusters
// @param basis_degree vector containing the desired basis degree for each dimension
// @param M Int containing the number of eigenfunctions
// @param boundary_knots matrix containing the boundary knots for each dimension (each row is a dimension)
// @param internal_knots field of vectors containing the internal knots for each dimension
// @param tot_mcmc_iters Int containing total number of MCMC iterations
// @param c Vector containing hyperparmeters for pi
// @param b double containing hyperparameter for alpha_3
// @param a_12 Vector containing hyperparameters for sampling from delta
// @param alpha1l Double containing hyperparameters for sampling from A
// @param alpha2l Double containing hyperparameters for sampling from A
// @param beta1l Double containing hyperparameters for sampling from A
// @param beta2l Double containing hyperparameters for sampling from A
// @param var_pi Double containing variance parameter of the random walk MH for pi parameter
// @param var_Z Double containing variance parameter of the random walk MH for Z parameter
// @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
// @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
// @param alpha Double containing hyperparameters for sampling from tau
// @param beta Double containing hyperparameters for sampling from tau
// @param alpha_0 Double containing hyperparameters for sampling from sigma
// @param beta_0 Double containing hyperparameters for sampling from sigma
// @param Z_est Matrix containing Z values to be conditioned on
// @param nu_est Matrix containing nu values to be conditioned on
// @returns params List of objects containing the MCMC samples from the last batch
inline Rcpp::List BMFPMM_Theta(const arma::field<arma::vec>& y_obs,
                               const arma::field<arma::mat>& t_obs,
                               const int& n_funct,
                               const int& K,
                               const arma::vec& basis_degree,
                               const int& M,
                               const arma::mat& boundary_knots,
                               const arma::field<arma::vec>& internal_knots,
                               const int& tot_mcmc_iters,
                               const arma::vec& c,
                               const double& b,
                               const double& nu_1,
                               const double& alpha1l,
                               const double& alpha2l,
                               const double& beta1l,
                               const double& beta2l,
                               const double& a_Z_PM,
                               const double& a_pi_PM,
                               const double& var_alpha3,
                               const double& var_epsilon1,
                               const double& var_epsilon2,
                               const double& alpha,
                               const double& beta,
                               const double& alpha_0,
                               const double& beta_0,
                               const arma::mat& Z_est,
                               const arma::mat& nu_est){
  // Make B_obs
  arma::field<arma::mat> B_obs = TensorBSpline(t_obs, n_funct, basis_degree,
                                               boundary_knots, internal_knots);

  arma::mat P_mat = GetP(basis_degree,internal_knots);

  int P = P_mat.n_cols;
  arma::cube nu(K, P, tot_mcmc_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, tot_mcmc_iters, arma::fill::randn);
  arma::mat pi(K, tot_mcmc_iters, arma::fill::zeros);
  arma::vec pi_ph = arma::zeros(K);
  pi.col(0) = rdirichlet(c);
  arma::vec sigma(tot_mcmc_iters, arma::fill::ones);
  arma::vec Z_ph = arma::zeros(K);
  arma::vec alpha_3 = arma::ones(tot_mcmc_iters);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, tot_mcmc_iters,
                                         arma::distr_param(0,1));

  for(int i = 0; i < n_funct; i++){
    Z.slice(0).row(i) = rdirichlet(pi.col(0)).t();
  }

  arma::mat delta(M, tot_mcmc_iters, arma::fill::ones);
  arma::field<arma::cube> gamma(tot_mcmc_iters,1);
  arma::field<arma::cube> Phi(tot_mcmc_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(tot_mcmc_iters, 2);
  arma::vec loglik = arma::zeros(tot_mcmc_iters);

  for(int i = 0; i < tot_mcmc_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(tot_mcmc_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  Z.slice(0) = Z_est;
  nu.slice(0) = nu_est;

  for(int i = 1; i < tot_mcmc_iters; i++){
    Z.slice(i) = Z_est;
    nu.slice(i) = nu_est;
  }


  for(int i = 0; i < tot_mcmc_iters; i++){
    tilde_tau(0) = delta(0, (i));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i));
    }

    updatePhi(y_obs, B_obs, nu.slice((i)),
              gamma((i),0), tilde_tau,
              Z.slice((i)), chi.slice((i)),
              sigma((i)), (i),
              tot_mcmc_iters, m_1, M_1, Phi);

    updateDelta(Phi((i),0), gamma((i),0),
                A.row(i).t(), (i),
                tot_mcmc_iters, delta);

    updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i)),
            var_epsilon1, var_epsilon2, (i), tot_mcmc_iters, A);

    updateGamma(nu_1, delta.col((i)), Phi((i),0),
                (i), tot_mcmc_iters, gamma);

    updateTau(alpha, beta, nu.slice((i)), (i),
              tot_mcmc_iters, P_mat, tau);

    updateSigma(y_obs, B_obs, alpha_0, beta_0,
                nu.slice((i)), Phi((i),0),
                Z.slice((i)), chi.slice((i)),
                (i), tot_mcmc_iters, sigma);

    updateChi(y_obs, B_obs, Phi((i),0),
              nu.slice((i)), Z.slice((i)),
              sigma((i)), (i), tot_mcmc_iters,
              chi);

    // Calculate log likelihood
    loglik((i)) =  calcLikelihood(y_obs, B_obs, nu.slice((i)),
           Phi((i),0), Z.slice((i)), chi.slice((i)), sigma((i)));
    if(((i+1) % 100) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i)-99, (i))) << "\n";
      Rcpp::checkUserInterrupt();
    }
  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("Z", Z),
                                         Rcpp::Named("nu", nu),
                                         Rcpp::Named("chi", chi),
                                         Rcpp::Named("A", A),
                                         Rcpp::Named("delta", delta),
                                         Rcpp::Named("sigma", sigma),
                                         Rcpp::Named("tau", tau),
                                         Rcpp::Named("gamma", gamma),
                                         Rcpp::Named("Phi", Phi),
                                         Rcpp::Named("loglik", loglik));
  return params;
}


}

#endif
