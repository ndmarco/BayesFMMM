#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "Distributions.H"
#include "UpdateClassMembership.H"
#include "UpdatePartialMembership.H"
#include "UpdatePi.H"
#include "UpdatePhi.H"
#include "UpdateDelta.H"
#include "UpdateA.H"
#include "UpdateGamma.H"
#include "UpdateNu.H"
#include "UpdateTau.H"
#include "UpdateSigma.H"
#include "UpdateChi.H"
#include "UpdateYStar.H"
#include "BFPMM.H"
#include "EstimateInitialState.H"
#include "UpdateAlpha3.H"
#include "LabelSwitch.H"

//' Function for finding a good initial starting point for nu parameters and Z parameters, with option for temperered transitions
//'
//' @name BFPMM_Nu_Z_multiple_try
//' @param tot_mcmc_iters Int conatining the number of MCMC iterations per try
//' @param beta_N_t Double containing the maximum weight for tempered transisitons
//' @param N_t Int containing total number of tempered transitions. If no tempered transitions are desired, pick a small integer
//' @param n_temp_trans Int containing how often tempered transitions are performed. If no tempered transitions are desired, pick a integer larger than tot_mcmc_iters
//' @param n_try Int containing how many different chains are tried
//' @param k Int containing the number of clusters
//' @param Y Field of vectors containing the observed values
//' @param time Field of vecotrs containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param n_basis Int containing the number of basis functions
//' @param n_eigen Int containing the number of eigenfunctions
//' @returns BestChain List containing a summary of the best performing chain
//' @export
// [[Rcpp::export]]
Rcpp::List BFPMM_Nu_Z_multiple_try(const int tot_mcmc_iters,
                                   const double beta_N_t,
                                   const int N_t,
                                   const int n_temp_trans,
                                   const int n_try,
                                   const int k,
                                   const arma::field<arma::vec> Y,
                                   const arma::field<arma::vec> time,
                                   const int n_funct,
                                   const int n_basis,
                                   const int n_eigen){
  splines2::BSpline bspline;
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  for(int i = 0; i < n_funct; i++)
  {
    // Create Bspline object with n_basis degrees of freedom
    // n_basis - 3 - 1 internal nodes
    bspline = splines2::BSpline(time(i,0), n_basis);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat{bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  // placeholder
  arma::field<arma::vec> t_star1(n_funct,1);
  arma::vec c = arma::ones(k);

  // start MCMC sampling
  Rcpp::List mod1 = BFPMM_Nu_Z(Y, time, n_funct, k, n_basis, n_eigen, tot_mcmc_iters,
                               n_temp_trans, t_star1, c, 800, 3, 2, 3, 1, 1,
                               1000, 1000, 0.05, sqrt(1), sqrt(1), 1, 10, 1, 1, beta_N_t,
                               N_t);
  arma::vec ph = mod1["loglik"];
  double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

  for(int i = 0; i < n_try; i++){
    Rcpp::List modi = BFPMM_Nu_Z(Y, time, n_funct, k, n_basis, n_eigen, tot_mcmc_iters,
                                 n_temp_trans, t_star1, c, 800, 3, 2, 3, 1, 1,
                                 1000, 1000, 0.05, sqrt(1), sqrt(1), 1, 10, 1, 1, beta_N_t,
                                 N_t);
    arma::vec ph1 = modi["loglik"];
    if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
      mod1 = modi;
      min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
    }

  }

  Rcpp::List BestChain =  Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                             Rcpp::Named("pi", mod1["pi"]),
                                             Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                             Rcpp::Named("A", mod1["A"]),
                                             Rcpp::Named("delta", mod1["delta"]),
                                             Rcpp::Named("sigma", mod1["sigma"]),
                                             Rcpp::Named("tau", mod1["tau"]),
                                             Rcpp::Named("Z", mod1["Z"]),
                                             Rcpp::Named("loglik", mod1["loglik"]));

  return BestChain;
}

//' Estimates the initial starting point of the rest of the parameters given an intial starting point for Z and nu
//'
//' @name BFPMM_Theta_Est
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param Z_samp Cube containing initial chain of Z parameters from BFPMM_Nu_Z_multiple_try
//' @param nu_samp Cube containing intial chain of nu paramaeters from BFPMM_Nu_Z_multiple_try
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param k Int containing the number of clusters
//' @param Y Field of vectors containing the observed values
//' @param time Field of vecotrs containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param n_basis Int containing the number of basis functions
//' @param n_eigen Int containing the number of eigenfunctions
//' @returns BestChain List containing a summary of the chain conditioned on nu and Z
//' @export
// [[Rcpp::export]]
Rcpp::List BFPMM_Theta_Est(const int tot_mcmc_iters,
                           const arma::cube Z_samp,
                           const arma::cube nu_samp,
                           double burnin_prop,
                           const int k,
                           const arma::field<arma::vec> Y,
                           const arma::field<arma::vec> time,
                           const int n_funct,
                           const int n_basis,
                           const int n_eigen){

  splines2::BSpline bspline;
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  for(int i = 0; i < n_funct; i++)
  {
    // Create Bspline object with n_basis degrees of freedom
    // n_basis - 3 - 1 internal nodes
    bspline = splines2::BSpline(time(i,0), n_basis);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat{bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  // placeholder
  arma::field<arma::vec> t_star1(n_funct,1);

  arma::vec c = arma::ones(k);

  int n_nu = nu_samp.n_slices;
  arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
  arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
  arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  for(int i = 0; i < Z_est.n_cols; i++){
    for(int j = 0; j < Z_est.n_rows; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
      }
      Z_est(j,i) = arma::median(ph_Z);
    }
    for(int j = 0; j < nu_samp.n_cols; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
      }
      nu_est(i,j) = arma::median(ph_nu);
    }
  }

  // start MCMC sampling
  Rcpp::List mod1 = BFPMM_Theta(Y, time, n_funct, k, n_basis, n_eigen, tot_mcmc_iters,
                                t_star1, c, 1, 3, 2, 3, 1, 1, 1000, 1000, 0.05,
                                sqrt(1), sqrt(1), 1, 5, 1, 1, Z_est, nu_est);

  Rcpp::List BestChain =  Rcpp::List::create(Rcpp::Named("chi", mod1["chi"]),
                                             Rcpp::Named("A", mod1["A"]),
                                             Rcpp::Named("delta", mod1["delta"]),
                                             Rcpp::Named("sigma", mod1["sigma"]),
                                             Rcpp::Named("tau", mod1["tau"]),
                                             Rcpp::Named("gamma", mod1["gamma"]),
                                             Rcpp::Named("Phi", mod1["Phi"]),
                                             Rcpp::Named("Nu_est", nu_est),
                                             Rcpp::Named("loglik", mod1["loglik"]));

  return BestChain;
}

//' Performs MCMC, with optional tempered transitions, using user specified starting points.
//'
//' @name BFPMM_warm_start
//' @param beta_N_t Double containing the maximum weight for tempered transisitons
//' @param N_t Int containing total number of tempered transitions. If no tempered transitions are desired, pick a small integer
//' @param n_temp_trans Int containing how often tempered transitions are performed. If no tempered transitions are desired, pick a integer larger than tot_mcmc_iters
//' @param tot_mcmc_iters Int conatining the number of MCMC iterations
//' @param r_stored_iters Int containing number of MCMC iterations stored in memory before writing to directory
//' @param Z_samp Cube containing initial chain of Z parameters
//' @param pi_samp Matrix containing intial chain of pi parameters
//' @param alpha_3_samp Vector containing intial chain of alpha_3 parameters
//' @param delta_samp Matrix containing initial chain of delta parameters
//' @param gamma_samp Field of cubes containing initial chain of gamma parameters
//' @param Phi_samp Field of cubes containing initial chain of phi parameters
//' @param A_samp Matrix containing intial chain of A parameters
//' @param nu_samp Cube containing intial chain of nu paramaeters
//' @param tau_samp Matrix containing initial chain of tau parameters
//' @param sigma_samp Vector containing initial chain of sigma parameters
//' @param chi_samp Cube containing initial chain of chi parameters
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param k Int containing the number of clusters
//' @param Y Field of vectors containing the observed values
//' @param time Field of vecotrs containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param n_basis Int containing the number of basis functions
//' @param n_eigen Int containing the number of eigenfunctions
//' @param thinning_num Int containing how often we should save MCMC iterations. Should be a divisible by r_stored_iters and tot_mcmc_iters
//' @param dir String containing directory where the MCMC files should be saved
//' @export
// [[Rcpp::export]]
Rcpp::List BFPMM_warm_start(const double beta_N_t,
                            const int N_t,
                            const int n_temp_trans,
                            const int tot_mcmc_iters,
                            const int r_stored_iters,
                            const arma::cube Z_samp,
                            const arma::mat pi_samp,
                            const arma::vec alpha_3_samp,
                            const arma::mat delta_samp,
                            const arma::field<arma::cube> gamma_samp,
                            const arma::field<arma::cube> Phi_samp,
                            const arma::mat A_samp,
                            const arma::cube nu_samp,
                            const arma::mat tau_samp,
                            const arma::vec sigma_samp,
                            const arma::cube chi_samp,
                            const double burnin_prop,
                            const int k,
                            const arma::field<arma::vec> Y,
                            const arma::field<arma::vec> time,
                            const int n_funct,
                            const int n_basis,
                            const int n_eigen,
                            const double thinning_num,
                            const std::string dir){
  splines2::BSpline bspline;
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  for(int i = 0; i < n_funct; i++)
  {
    // Create Bspline object with n_basis degrees of freedom
    // n_basis - 3 - 1 internal nodes
    bspline = splines2::BSpline(time(i,0), n_basis);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat{bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  // placeholder
  arma::field<arma::vec> t_star1(n_funct,1);

  arma::vec c = arma::ones(k);

  int n_nu = alpha_3_samp.n_elem;

  double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
  arma::vec pi_est = arma::zeros(pi_samp.n_rows);
  arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
  arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
  arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  for(int i = 0; i < Z_est.n_cols; i++){
    pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    for(int j = 0; j < Z_est.n_rows; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
      }
      Z_est(j,i) = arma::median(ph_Z);
    }
    for(int j = 0; j < nu_samp.n_cols; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
      }
      nu_est(i,j) = arma::median(ph_nu);
    }
  }

  int n_Phi = sigma_samp.n_elem;

  double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
  arma::vec delta_est = arma::zeros(delta_samp.n_rows);
  arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  for(int i = 0; i < delta_samp.n_rows; i++){
    for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
      ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,l);
    }
    delta_est(i) = arma::median(ph_delta);
  }
  arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
  arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
  arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  for(int i = 0; i < Phi_est.n_rows; i++){
    for(int j = 0; j < Phi_est.n_cols; j++){
      for(int m = 0; m < Phi_est.n_slices; m++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);

          ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
        }
        Phi_est(i,j,m) = arma::median(ph_phi);
        gamma_est(i,j,m) = arma::median(ph_gamma);
      }
    }
  }

  arma::vec A_est = arma::zeros(A_samp.n_cols);
  arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  for(int i = 0; i < A_est.n_elem; i++){
    for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
      ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(l,i);
    }
    A_est(i) = arma::median(ph_A);
  }
  arma::vec tau_est = arma::zeros(tau_samp.n_cols);
  arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  for(int i = 0; i < tau_est.n_elem; i++){
    for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
      ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
    }
    tau_est(i) = arma::median(ph_tau);
  }
  arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
  arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  for(int i = 0; i < chi_est.n_rows; i++){
    for(int j = 0; j < chi_est.n_cols; j++){
      for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
        ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
      }
      chi_est(i,j) = arma::median(ph_chi);
    }
  }

  // start MCMC sampling
  Rcpp::List mod1 = BFPMM_MTT_warm_start(Y, time, n_funct, thinning_num, k,
                                         n_basis, n_eigen, tot_mcmc_iters,
                                         r_stored_iters, n_temp_trans, t_star1,
                                         c, 800, 3, 2, 3, 1, 1, 1000, 1000, 0.05,
                                         sqrt(1), sqrt(1), 1, 10, 1, 1, dir,
                                         beta_N_t, N_t, Z_est, pi_est, alpha_3_est,
                                         delta_est, gamma_est, Phi_est, A_est,
                                         nu_est, tau_est, sigma_est, chi_est);

  Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                        Rcpp::Named("chi", mod1["chi"]),
                                        Rcpp::Named("pi", mod1["pi"]),
                                        Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                        Rcpp::Named("A", mod1["A"]),
                                        Rcpp::Named("delta", mod1["delta"]),
                                        Rcpp::Named("sigma", mod1["sigma"]),
                                        Rcpp::Named("tau", mod1["tau"]),
                                        Rcpp::Named("gamma", mod1["gamma"]),
                                        Rcpp::Named("Phi", mod1["Phi"]),
                                        Rcpp::Named("Z", mod1["Z"]),
                                        Rcpp::Named("loglik", mod1["loglik"]));

  return mod2;
}
