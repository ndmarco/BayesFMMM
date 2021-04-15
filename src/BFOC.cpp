#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "Distributions.H"
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

//' Run the Bayesian Functional Overlapping Clusters model
//'
//' @name BFOC
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
Rcpp::List BFOC(const arma::field<arma::vec> y_obs,
                const arma::field<arma::vec> t_obs,
                const double n_funct,
                const int K,
                const int P,
                const int M,
                const int tot_mcmc_iters,
                const arma::field<arma::vec> t_star,
                const double nu_1,
                const double rho,
                const double alpha_3,
                const double alpha1l,
                const double alpha2l,
                const double beta1l,
                const double beta2l,
                const double var_epsilon1,
                const double var_epsilon2,
                const double alpha,
                const double beta,
                const double alpha_0,
                const double beta_0){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);

  arma::field<arma::mat> B_star(n_funct,1);
  arma::field<arma::mat> y_star(tot_mcmc_iters, 1);
  arma::field<arma::vec> t_comb(n_funct,1);

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
      y_star(i,0) = arma::randn(tot_mcmc_iters, t_star(i,0).n_elem);
    }else{
      splines2::BSpline bspline;
      // Create Bspline object with 8 degrees of freedom
      // 8 - 3 - 1 internal nodes
      bspline = splines2::BSpline(t_obs(i,0), P);
      // Get Basis matrix (100 x 8)
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

  arma::cube nu(K, P, tot_mcmc_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, tot_mcmc_iters, arma::fill::randn);
  arma::mat pi(K, tot_mcmc_iters, arma::fill::zeros);
  arma::vec sigma(tot_mcmc_iters, arma::fill::ones);
  arma::mat Z_ph(n_funct, K, arma::fill::ones);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, tot_mcmc_iters,
                                         arma::distr_param(0,1));
  arma::mat delta(M, tot_mcmc_iters, arma::fill::ones);

  arma::field<arma::cube> gamma(tot_mcmc_iters,1);
  arma::field<arma::cube> Phi(tot_mcmc_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(tot_mcmc_iters, 2);
  arma::vec loglik = arma::zeros(tot_mcmc_iters);

  for(int i = 0; i < tot_mcmc_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::cube(K, P, M, arma::fill::ones);
  }

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(tot_mcmc_iters, K, arma::fill::ones);

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  for(int i = 0; i < tot_mcmc_iters; i++){
    updateZ(y_obs, y_star, B_obs, B_star, Phi(i,0), nu.slice(i), chi.slice(i),
            pi.col(i), sigma(i), rho, i, tot_mcmc_iters, Z_ph, Z);
    updatePi(alpha_3, Z.slice(i), i, tot_mcmc_iters,  pi);

    tilde_tau(0) = delta(0, i);
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,i);
    }
    updatePhi(y_obs, y_star, B_obs, B_star, nu.slice(i), gamma(i,0),
              tilde_tau, Z.slice(i), chi.slice(i), sigma(i), i,
              tot_mcmc_iters, m_1, M_1, Phi);
    updateDelta(Phi(i,0), gamma(i,0), A.row(i).t(), i, tot_mcmc_iters,
                delta);
    updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col(i), var_epsilon1,
            var_epsilon2, i, tot_mcmc_iters, A);
    updateGamma(nu_1, delta.col(i), Phi(i,0), i, tot_mcmc_iters, gamma);
    updateNu(y_obs, y_star, B_obs, B_star, tau.row(i).t(), Phi(i,0), Z.slice(i),
             chi.slice(i), sigma(i), i, tot_mcmc_iters, P_mat, b_1, B_1, nu);
    updateTau(alpha, beta, nu.slice(i), i, tot_mcmc_iters, P_mat, tau);
    updateSigma(y_obs, y_star, B_obs, B_star, alpha_0, beta_0, nu.slice(i),
                Phi(i,0), Z.slice(i), chi.slice(i), i, tot_mcmc_iters, sigma);
    updateChi(y_obs, y_star, B_obs, B_star, Phi(i,0), nu.slice(i), Z.slice(i),
               sigma(i), i, tot_mcmc_iters, chi);
    // updateYStar(B_star, nu.slice(i), Phi(i,0), Z.slice(i), chi.slice(i),
    //             sigma(i), i, tot_mcmc_iters, y_star);

    // Calculate log likelihood
    loglik(i) =  calcLikelihood(y_obs, y_star, B_obs, B_star, nu.slice(i),
           Phi(i,0), Z.slice(i), chi.slice(i),i, sigma(i));
    if(((i+1) % 20) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec(i-19, i)) << "\n";
      Rcpp::checkUserInterrupt();
    }
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu", nu),
                                      Rcpp::Named("y_star", y_star),
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
  return mod;
}

//'
//' @name BFOC
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
Rcpp::List BFOC_SS(const arma::mat known_Z,
                   const arma::field<arma::vec> y_obs,
                   const arma::field<arma::vec> t_obs,
                   const double n_funct,
                   const int K,
                   const int P,
                   const int M,
                   const int tot_mcmc_iters,
                   const int r_stored_iters,
                   const arma::field<arma::vec> t_star,
                   const double nu_1,
                   const double rho,
                   const double alpha_3,
                   const double alpha1l,
                   const double alpha2l,
                   const double beta1l,
                   const double beta2l,
                   const double var_epsilon1,
                   const double var_epsilon2,
                   const double alpha,
                   const double beta,
                   const double alpha_0,
                   const double beta_0,
                   const std::string directory){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  arma::field<arma::mat> B_star(n_funct,1);
  arma::field<arma::mat> y_star(r_stored_iters, 1);
  arma::field<arma::vec> t_comb(n_funct,1);

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
      // Create Bspline object with 8 degrees of freedom
      // 8 - 3 - 1 internal nodes
      bspline = splines2::BSpline(t_obs(i,0), P);
      // Get Basis matrix (100 x 8)
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

  arma::cube nu(K, P, r_stored_iters, arma::fill::randn);
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  arma::mat Z_ph(n_funct, K, arma::fill::ones);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));

  Z.slice(0).submat(0, 0, known_Z.n_rows - 1, K-1) = known_Z;
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
    updateZ(y_obs, y_star, B_obs, B_star, Phi((i % r_stored_iters),0), nu.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
            pi.col((i % r_stored_iters)), sigma((i % r_stored_iters)), rho, known_Z.n_rows, (i % r_stored_iters), r_stored_iters, Z_ph, Z);
    updatePi(alpha_3, Z.slice((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters,  pi);

    tilde_tau(0) = delta(0, (i % r_stored_iters));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i % r_stored_iters));
    }
    updatePhi(y_obs, y_star, B_obs, B_star, nu.slice((i % r_stored_iters)), gamma((i % r_stored_iters),0),
              tilde_tau, Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)), (i % r_stored_iters),
              r_stored_iters, m_1, M_1, Phi);
    updateDelta(Phi((i % r_stored_iters),0), gamma((i % r_stored_iters),0), A.row(i % r_stored_iters).t(), (i % r_stored_iters), r_stored_iters,
                delta);
    updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i % r_stored_iters)), var_epsilon1,
            var_epsilon2, (i % r_stored_iters), r_stored_iters, A);
    updateGamma(nu_1, delta.col((i % r_stored_iters)), Phi((i % r_stored_iters),0), (i % r_stored_iters), r_stored_iters, gamma);
    updateNu(y_obs, y_star, B_obs, B_star, tau.row((i % r_stored_iters)).t(), Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)),
             chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, P_mat, b_1, B_1, nu);
    updateTau(alpha, beta, nu.slice((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, P_mat, tau);
    updateSigma(y_obs, y_star, B_obs, B_star, alpha_0, beta_0, nu.slice((i % r_stored_iters)),
                Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, sigma);
    updateChi(y_obs, y_star, B_obs, B_star, Phi((i % r_stored_iters),0), nu.slice((i % r_stored_iters)), Z.slice((i % r_stored_iters)),
              sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, chi);
    updateYStar(B_star, nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
                sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, y_star);

    // Calculate log likelihood
    loglik((i % r_stored_iters)) =  calcLikelihood(y_obs, y_star, B_obs, B_star, nu.slice((i % r_stored_iters)),
           Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),(i % r_stored_iters), sigma((i % r_stored_iters)));
    if(((i+1) % 20) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i % r_stored_iters)-19, (i % r_stored_iters))) << "\n";
      Rcpp::checkUserInterrupt();
    }
    if(((i+1) % r_stored_iters) == 0 && i > 1){
      // Save parameters
      nu.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
      y_star.save(directory + "Y_Star" + std::to_string(q) + ".txt");
      chi.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
      pi.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
      A.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
      delta.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
      sigma.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
      tau.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
      gamma.save(directory + "Gamma" + std::to_string(q) +".txt");
      Phi.save(directory + "Phi" + std::to_string(q) +".txt");
      Z.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

      //reset all parameters
      nu.slice(0) = nu.slice(i % r_stored_iters);
      for(int b = 0; b < n_funct; b++){
        if(y_star(b,0).n_elem > 0){
          y_star(b,0).row(0) =  y_star(b,0).row(i % r_stored_iters);
        }
      }
      chi.slice(0) = chi.slice(i % r_stored_iters);
      pi.col(0) = pi.col(i % r_stored_iters);
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
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu", nu),
                                      Rcpp::Named("y_star", y_star),
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
  return mod;
}
