#include <RcppArmadillo.h>
#include<cmath>
#include <splines2Armadillo.h>
#include "UpdateClassMembership.H"
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
#include "CalculateLikelihood.H"

arma::mat BasisExpansion(const arma::field<arma::vec>& y_obs,
                         const arma::field<arma::mat>& B_obs,
                         const int& n_funct,
                         const int& K,
                         const int& P){
  arma::mat theta(P, n_funct, arma::fill::zeros);
  for(int i = 0; i < n_funct; i++ ){
    theta.col(i) = arma::solve((B_obs(i,0).t() * B_obs(i,0)),
                B_obs(i,0).t() * y_obs(i,0));
  }

  return theta;
}

double SigmaInitialState(const arma::field<arma::vec>& y_obs,
                         const arma::field<arma::mat>& B_obs,
                         const arma::mat& theta,
                         const int& n_funct){
  double sigma = 0;
  double numerator = 0;
  double denominator = 0;
  for(int i = 0; i < n_funct; i++){
    for(int j = 0; j < B_obs(i,0).n_rows; j++){
      numerator = numerator + std::pow(y_obs(i,0)(j) - arma::dot(B_obs(i,0).row(j),
        theta.col(i)), 2);
    }
    denominator = denominator + y_obs(i,0).n_elem;
  }
  sigma = numerator / denominator;
  return sigma;
}

arma::mat NuInitialState(const arma::field<arma::mat>& B_obs,
                         const arma::mat& Z_known,
                         const arma::mat& theta,
                         const int& n_funct){
  arma::mat ph(theta.n_rows, Z_known.n_cols, arma::fill::zeros);
  arma::mat ph1(Z_known.n_cols, Z_known.n_cols, arma::fill::zeros);
  arma::mat nu(Z_known.n_cols, theta.n_rows, arma::fill::zeros);
  for(int i = 0; i < n_funct; i++){
    ph = ph + (theta.col(i) * Z_known.row(i));
    ph1 = ph1 + (Z_known.row(i).t() * Z_known.row(i));
  }
  nu = (ph * arma::pinv(ph1));
  nu = nu.t();
  return nu;
}

void GetDistance(const arma::vec& means,
                 const arma::mat& theta,
                 const arma::field<arma::mat>& B_obs,
                 arma::vec& dist){
  dist.zeros();
  for(int i = 0; i < dist.n_rows; i++){
    for(int k = 0; k < B_obs(i,0).n_rows; k++){
      dist(i) = dist(i) + std::pow(arma::dot(B_obs(i,0).row(k),
        (theta.col(i) - means)), 2);
    }
      dist(i) = dist(i) / B_obs(i,0).n_rows;
  }
}

arma::mat ZInitialState(const arma::field<arma::mat>& B_obs,
                        const arma::mat& theta,
                        const double& alpha,
                        const int& max_iter,
                        const int& K,
                        const int& n_funct,
                        const double& convergence){
  arma::mat Z(n_funct, K, arma::fill::zeros);
  arma::vec Z_i(K, arma::fill::zeros);
  arma::vec dist(n_funct, arma::fill::zeros);
  arma::vec dist_min(n_funct, arma::fill::zeros);
  arma::mat means(theta.n_rows, K, arma::fill::randn);
  arma::mat means_last(theta.n_rows, K, arma::fill::randn);
  arma::vec means_ph(theta.n_rows, arma::fill::randn);
  int num = 0;
  bool converge = false;
  int i = 0;
  while(i < max_iter && converge == false){
    Z.zeros();
    for(int j = 1; j < (std::pow(2,K) + 1); j++){
      Z_i.zeros();
      num = j;
      for(int k = K-1; k >= 0; k--){
        if(std::pow(2, k) <= num){
          Z_i(k) = 1;
          num = num - std::pow(2, k);
        }
      }
      means_ph = means * Z_i;
      GetDistance(means_ph, theta, B_obs,  dist);
      if(j == 1){
        dist_min = dist;
        for(int m = 0; m < n_funct; m++){
          Z.row(m) = Z_i.t();
        }
      }
      if(j > 1){
        for(int m = 0; m < n_funct; m++){
          if(dist(m) < dist_min(m)){
            dist_min(m) = dist(m);
            Z.row(m) = Z_i.t();
          }
        }
      }
    }
    means_last = means;
    means = NuInitialState(B_obs, Z, theta, n_funct).t();
    Rcpp::Rcout << "Iteration: " << i+1 << "\n";
    Rcpp::Rcout << "Delta: " << arma::norm(means_last - means) << "\n";

    if(arma::norm(means_last - means) < convergence){
      converge = true;
    }
    i++;
  }

  return Z;
}

//'
//' @name GetPhiChi
//' @param y_obs Field (list) of vectors containing the observed values
//' @param t_obs Field (list) of vectors containing time points of observed values
//' @param n_funct Double containing number of functions observed
//' @param P Int that indicates the number of b-spline basis functions
//' @param M int that indicates the number of slices used in Phi parameter
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param t_star Field (list) of vectors containing time points of interest that are not observed (optional)
//' @param rho Double containing hyperparmater for sampling from Z
//' @param alpha_3 Double hyperparameter for sampling from pi
//' @param a_12 Vec containing hyperparameters for sampling from delta
//' @param alpha1l Double containing hyperparameters for sampling from A
//' @param alpha2l Double containing hyperparameters for sampling from A
//' @param beta1l Double containing hyperparameters for sampling from A
//' @param beta2l Double containing hyperparameters for sampling from A
//' @param var_epslion1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epslion2 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha Double containing hyperparameters for sampling from tau
//' @param beta Double containing hyperparameters for sampling from tau
//' @param alpha_0 Double containing hyperparameters for sampling from sigma
//' @param beta_0 Double containing hyperparameters for sampling from sigma
//' @export
// [[Rcpp::export]]
Rcpp::List PhiChiInitialState(const arma::mat& known_Z,
                              const arma::field<arma::vec>& y_obs,
                              const arma::field<arma::vec>& t_obs,
                              const double& n_funct,
                              const int& K,
                              const int& P,
                              const int& M,
                              const int& tot_mcmc_iters,
                              const int& r_stored_iters,
                              const arma::field<arma::vec>& t_star,
                              const double& nu_1,
                              const double& rho,
                              const double& alpha_3,
                              const double& alpha1l,
                              const double& alpha2l,
                              const double& beta1l,
                              const double& beta2l,
                              const double& var_epsilon1,
                              const double& var_epsilon2,
                              const double& alpha,
                              const double& beta,
                              const double& alpha_0,
                              const double& beta_0,
                              const arma::mat& nu,
                              const double& sigma){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  arma::field<arma::mat> B_star(n_funct,1);
  arma::field<arma::mat> y_star(n_funct, 1);
  arma::field<arma::vec> t_comb(n_funct,1);

  Rcpp::Rcout << "Starting MCMC to estimate inital starting points";

  for(int i = 0; i < n_funct; i++){
    if(t_star(i,0).n_elem > 0){
      t_comb(i,0) = arma::zeros(t_obs(i,0).n_elem + t_star(i,0).n_elem);
      t_comb(i,0).subvec(0, t_obs(i,0).n_elem - 1) = t_obs(i,0);
      t_comb(i,0).subvec(t_obs(i,0).n_elem, t_obs(i,0).n_elem + t_star(i,0).n_elem - 1)
        = t_star(i,0);
      splines2::BSpline bspline;
      // Create Bspline object with 8 degrees of freedom
      // 8 - 3 - 1 internal nodes
      bspline = splines2::BSpline(t_comb(i,0), P);
      // Get Basis matrix (100 x 8)
      arma::mat bspline_mat {bspline.basis(true)};

      B_obs(i,0) = bspline_mat.submat(0, 0, t_obs(i,0).n_elem - 1, P-1);
      B_star(i,0) =  bspline_mat.submat(t_obs(i,0).n_elem, 0,
             t_obs(i,0).n_elem + t_star(i,0).n_elem - 1, P-1);
      y_star(i,0) = arma::randn(r_stored_iters, t_star(i,0).n_elem);
    }else{
      splines2::BSpline bspline;
      // Create Bspline object with P degrees of freedom
      // P - 3 - 1 internal nodes
      bspline = splines2::BSpline(t_obs(i,0), P);
      // Get Basis matrix (n_funct x P)
      arma::mat bspline_mat {bspline.basis(true)};
      B_obs(i,0) = bspline_mat;
    }
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
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  arma::mat Z_ph(n_funct, K, arma::fill::ones);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));

  Z.slice(0).submat(0, 0, known_Z.n_rows - 1, K-1) = known_Z;
  arma::mat delta(M, r_stored_iters, arma::fill::ones);

  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);

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
    updateZ(y_obs, y_star, B_obs, B_star, Phi((i % r_stored_iters),0), nu, chi.slice((i % r_stored_iters)),
            pi.col((i % r_stored_iters)), sigma, rho, known_Z.n_rows, (i % r_stored_iters), r_stored_iters, Z_ph, Z);
    updatePi(alpha_3, Z.slice((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters,  pi);

    tilde_tau(0) = delta(0, (i % r_stored_iters));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i % r_stored_iters));
    }
    updatePhi(y_obs, y_star, B_obs, B_star, nu, gamma((i % r_stored_iters),0),
              tilde_tau, Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)), sigma, (i % r_stored_iters),
              r_stored_iters, m_1, M_1, Phi);
    updateDelta(Phi((i % r_stored_iters),0), gamma((i % r_stored_iters),0), A.row(i % r_stored_iters).t(), (i % r_stored_iters), r_stored_iters,
                delta);
    updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i % r_stored_iters)), var_epsilon1,
            var_epsilon2, (i % r_stored_iters), r_stored_iters, A);
    updateGamma(nu_1, delta.col((i % r_stored_iters)), Phi((i % r_stored_iters),0), (i % r_stored_iters), r_stored_iters, gamma);
    updateTau(alpha, beta, nu, (i % r_stored_iters), r_stored_iters, P_mat, tau);
    updateChi(y_obs, y_star, B_obs, B_star, Phi((i % r_stored_iters),0), nu, Z.slice((i % r_stored_iters)),
              sigma, (i % r_stored_iters), r_stored_iters, chi);
    updateYStar(B_star, nu, Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                sigma, (i % r_stored_iters), r_stored_iters, y_star);

    if(((i+1) % 20) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::checkUserInterrupt();
    }
    if(((i+1) % r_stored_iters) == 0 && i > 1){
      //reset all parameters
      for(int b = 0; b < n_funct; b++){
        if(y_star(b,0).n_elem > 0){
          y_star(b,0).row(0) =  y_star(b,0).row(i % r_stored_iters);
        }
      }
      chi.slice(0) = chi.slice(i % r_stored_iters);
      pi.col(0) = pi.col(i % r_stored_iters);
      A.row(0) = A.row(i % r_stored_iters);
      delta.col(0) = delta.col(i % r_stored_iters);
      tau.row(0) = tau.row(i % r_stored_iters);
      gamma(0,0) = gamma(i % r_stored_iters, 0);
      Phi(0,0) = Phi(i % r_stored_iters, 0);
      Z.slice(0) = Z.slice(i % r_stored_iters);

    }
  }

  // initialize matrices for starting point estimates
  arma::mat chi_est(n_funct, M, arma::fill::zeros);
  arma::field<arma::vec> y_star_est(n_funct, 1);
  arma::vec pi_est(K, arma::fill::zeros);
  arma::vec A_est(2, arma::fill::zeros);
  arma::vec delta_est(M, arma::fill::zeros);
  arma::cube gamma_est(K, P, M, arma::fill::zeros);
  arma::cube Phi_est(K, P, M, arma::fill::zeros);
  arma::vec tau_est(K, arma::fill::zeros);
  for(int k = 0; k < n_funct; k++){
    if(t_star(k,0).n_elem > 0){
      y_star_est(k,0) = arma::zeros(y_star(k,0).n_cols);
    }

  }

  for(int j = 0; j < r_stored_iters; j++){
    chi_est = chi_est + chi.slice(j);
    pi_est = pi_est + pi.col(j);
    A_est = A_est + A.row(j).t();
    delta_est = delta_est + delta.col(j);
    gamma_est = gamma_est + gamma(j,0);
    Phi_est = Phi_est + Phi(j,0);
    tau_est = tau_est + tau.row(j).t();
    for(int k = 0; k < n_funct; k++){
      if(t_star(k,0).n_elem > 0){
        y_star_est(k,0) =y_star_est(k,0) + y_star(k,0).row(j).t();
      }
    }
  }

  for(int k = 0; k < n_funct; k++){
    if(t_star(k,0).n_elem > 0){
      y_star_est(k,0) = y_star_est(k,0) / r_stored_iters;
    }
  }
  chi_est = chi_est / r_stored_iters;
  pi_est = pi_est / r_stored_iters;
  A_est = A_est / r_stored_iters;
  delta_est = delta_est / r_stored_iters;
  gamma_est = gamma_est / r_stored_iters;
  Phi_est = Phi_est / r_stored_iters;
  tau_est = tau_est / r_stored_iters;

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("y_star_est", y_star_est),
                                      Rcpp::Named("chi_est", chi_est),
                                      Rcpp::Named("pi_est", pi_est),
                                      Rcpp::Named("A_est", A_est),
                                      Rcpp::Named("delta_est", delta_est),
                                      Rcpp::Named("tau_est", tau_est),
                                      Rcpp::Named("gamma_est", gamma_est),
                                      Rcpp::Named("Phi_est", Phi_est));
  return mod;
}
