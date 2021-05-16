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
#include "CalculateTTAcceptance.H"

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
  arma::field<arma::mat> y_star(n_funct, 1);
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
//' @name BFOC_SS
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
  arma::field<arma::mat> y_star(n_funct, 1);
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
      arma::cube nu1(K, P, r_stored_iters/50, arma::fill::randn);
      arma::cube chi1(n_funct, M, r_stored_iters/50, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters/50, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters/50, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters/50, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_funct, K, r_stored_iters/50,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters/50, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters/50,1);
      arma::field<arma::cube> Phi1(r_stored_iters/50, 1);
      arma::mat tau1(r_stored_iters/50, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      arma::field<arma::mat> y_star1(r_stored_iters, 1);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);
      for(int j = 0; j < n_funct; j++){
        y_star1(j,0) = arma::randn(r_stored_iters/50, t_star(j,0).n_elem);
        y_star1(j,0).row(0) = y_star(j,0).row(0);
      }

      for(int p=1; p < r_stored_iters / 50; p++){
        nu1.slice(p) = nu.slice(50*p - 1);
        chi1.slice(p) = chi.slice(50*p - 1);
        pi1.col(p) = pi.col(50*p - 1);
        sigma1(p) = sigma(50*p - 1);
        A1.row(p) = A.row(50*p - 1);
        Z1.slice(p) = Z.slice(50*p - 1);
        delta1.col(p) = delta.col(p);
        gamma1(p,0) = gamma(50*p - 1,0);
        Phi1(p,0) = Phi(50*p  - 1,0);
        tau1.row(p) = tau.row(50*p - 1);
        for(int j = 0; j < n_funct; j++){
          y_star1(j,0).row(p) = y_star(j,0).row(50*p - 1);
        }
      }

      nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
      y_star1.save(directory + "Y_Star" + std::to_string(q) + ".txt");
      chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
      pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
      A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
      delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
      sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
      tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
      gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
      Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
      Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

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


//' Conducts un-tempered MCMC to estimate the posterior distribution in an unsupervised setting. MCMC samples will be stored in batches to a specified path.
//'
//' @name BFOC_U
//' @param y_obs Field (list) of vectors containing the observed values
//' @param t_obs Field (list) of vectors containing time points of observed values
//' @param n_funct Double containing number of functions observed
//' @param P Int that indicates the number of b-spline basis functions
//' @param M int that indicates the number of slices used in Phi parameter
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param r_stored_iters Int constaining number of iterations performed for each batch
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
//' @param directory String containing path to store batches of MCMC samples
//' @param Z_est Matrix containing initial starting point of Z matrix
//' @param A_est Vector containing initial starting point of A vector
//' @param pi_est Vector containing initial starting point of pi vector
//' @param tau_est Vector containing initial starting point of tau vector
//' @param delta_est Vector containing initial starting point of delta vector
//' @param nu_est Matrix containing initial starting point of nu matrix
//' @param Phi_est Cube containing initial starting point of Phi matrix
//' @param gamma_est Cube containing initial starting point of gamma matrix
//' @param chi_est Matrix containing initial starting point of chi matrix
//' @param y_star_est Field of Vectors containing initial starting point of y_star
//' @param sigma_est Double containing starting point of sigma parameter
//' @returns params List of objects containing the MCMC samples from the last batch
//' @export
// [[Rcpp::export]]
Rcpp::List BFOC_U(const arma::field<arma::vec>& y_obs,
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
                  const std::string directory,
                  const arma::mat& Z_est,
                  const arma::vec& A_est,
                  const arma::vec& pi_est,
                  const arma::vec& tau_est,
                  const arma::vec& delta_est,
                  const arma::mat& nu_est,
                  const arma::cube& Phi_est,
                  const arma::cube& gamma_est,
                  const arma::mat& chi_est,
                  const arma::field<arma::vec>& y_star_est,
                  const double& sigma_est){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  arma::field<arma::mat> B_star(n_funct,1);
  arma::field<arma::mat> y_star(n_funct, 1);
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
      y_star(i,0).row(0) = y_star_est(i,0).t();
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
  nu.slice(0) = nu_est;
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  chi.slice(0) = chi_est;
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  pi.col(0) = pi_est;
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  sigma(0) = sigma_est;
  arma::mat Z_ph(n_funct, K, arma::fill::ones);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));

  Z.slice(0) = Z_est;
  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  delta.col(0) = delta_est;
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  A.row(0) = A_est.t();
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }
  gamma(0,0) = gamma_est;
  Phi(0,0) = Phi_est;

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);
  tau.row(0) = tau_est.t();

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  for(int i = 0; i < tot_mcmc_iters; i++){
    updateZ(y_obs, y_star, B_obs, B_star, Phi((i % r_stored_iters),0), nu.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
            pi.col((i % r_stored_iters)), sigma((i % r_stored_iters)), rho, (i % r_stored_iters), r_stored_iters, Z_ph, Z);
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
      arma::cube nu1(K, P, r_stored_iters/50, arma::fill::randn);
      arma::cube chi1(n_funct, M, r_stored_iters/50, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters/50, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters/50, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters/50, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_funct, K, r_stored_iters/50,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters/50, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters/50,1);
      arma::field<arma::cube> Phi1(r_stored_iters/50, 1);
      arma::mat tau1(r_stored_iters/50, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      arma::field<arma::mat> y_star1(n_funct, 1);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);
      for(int j = 0; j < n_funct; j++){
        y_star1(j,0) = arma::randn(r_stored_iters/50, t_star(j,0).n_elem);
        y_star1(j,0).row(0) = y_star(j,0).row(0);
      }

      for(int p=1; p < r_stored_iters / 50; p++){
        nu1.slice(p) = nu.slice(50*p - 1);
        chi1.slice(p) = chi.slice(50*p - 1);
        pi1.col(p) = pi.col(50*p - 1);
        sigma1(p) = sigma(50*p - 1);
        A1.row(p) = A.row(50*p - 1);
        Z1.slice(p) = Z.slice(50*p - 1);
        delta1.col(p) = delta.col(p);
        gamma1(p,0) = gamma(50*p - 1,0);
        Phi1(p,0) = Phi(50*p  - 1,0);
        tau1.row(p) = tau.row(50*p - 1);
        for(int j = 0; j < n_funct; j++){
          y_star1(j,0).row(p) = y_star(j,0).row(50*p - 1);
        }
      }

      nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
      y_star1.save(directory + "Y_Star" + std::to_string(q) + ".txt");
      chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
      pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
      A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
      delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
      sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
      tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
      gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
      Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
      Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

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
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
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
  return params;
}

//' Conducts tempered MCMC to estimate the posterior distribution in an unsupervised setting. MCMC samples will be stored in batches to a specified path.
//'
//' @name BFOC_U_TT
//' @param y_obs Field (list) of vectors containing the observed values
//' @param t_obs Field (list) of vectors containing time points of observed values
//' @param n_funct Double containing number of functions observed
//' @param P Int that indicates the number of b-spline basis functions
//' @param M int that indicates the number of slices used in Phi parameter
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param r_stored_iters Int constaining number of iterations performed for each batch
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
//' @param directory String containing path to store batches of MCMC samples
//' @param Z_est Matrix containing initial starting point of Z matrix
//' @param A_est Vector containing initial starting point of A vector
//' @param pi_est Vector containing initial starting point of pi vector
//' @param tau_est Vector containing initial starting point of tau vector
//' @param delta_est Vector containing initial starting point of delta vector
//' @param nu_est Matrix containing initial starting point of nu matrix
//' @param Phi_est Cube containing initial starting point of Phi matrix
//' @param gamma_est Cube containing initial starting point of gamma matrix
//' @param chi_est Matrix containing initial starting point of chi matrix
//' @param y_star_est Field of Vectors containing initial starting point of y_star
//' @param sigma_est Double containing starting point of sigma parameter
//' @returns params List of objects containing the MCMC samples from the last batch
//' @export
// [[Rcpp::export]]
Rcpp::List BFOC_U_TT(const arma::field<arma::vec>& y_obs,
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
                     const std::string directory,
                     const arma::mat& Z_est,
                     const arma::vec& A_est,
                     const arma::vec& pi_est,
                     const arma::vec& tau_est,
                     const arma::vec& delta_est,
                     const arma::mat& nu_est,
                     const arma::cube& Phi_est,
                     const arma::cube& gamma_est,
                     const arma::mat& chi_est,
                     const arma::field<arma::vec>& y_star_est,
                     const double& beta_N_t,
                     const int& N_t,
                     const double& sigma_est){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  arma::field<arma::mat> B_star(n_funct,1);
  arma::field<arma::mat> y_star(n_funct, 1);
  arma::field<arma::mat> y_star_TT(n_funct, 1);
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

      // create placeholder for tempered transitions
      y_star_TT(i,0) = arma::randn((2 * N_t) + 1, t_star(i,0).n_elem);
      y_star(i,0).row(0) = y_star_est(i,0).t();
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
  nu.slice(0) = nu_est;
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  chi.slice(0) = chi_est;
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  pi.col(0) = pi_est;
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  sigma(0) = sigma_est;
  arma::mat Z_ph(n_funct, K, arma::fill::ones);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));
  Z.slice(0) = Z_est;
  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  delta.col(0) = delta_est;
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  A.row(0) = A_est.t();
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }
  gamma(0,0) = gamma_est;
  Phi(0,0) = Phi_est;

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);
  tau.row(0) = tau_est.t();

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

  for(int i = 0; i < ((2 * N_t) + 1); i++){
    gamma_TT(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi_TT(i,0) = arma::randn(K, P, M);
  }

  arma::mat tau_TT((2 * N_t) + 1, K, arma::fill::ones);

  int temp_ind = 0;
  double logA = 0;
  double logu = 0;
  int accept_num = 0;

  for(int i = 0; i < tot_mcmc_iters; i++){
    // initialize placeholders
    nu_TT.slice(0) = nu.slice(i);
    chi_TT.slice(0) = chi.slice(i);
    pi_TT.col(0) = pi.col(i);
    sigma_TT(0) = sigma(i);
    Z_TT.slice(0) = Z.slice(i);
    delta_TT.col(0) = delta.col(i);
    gamma_TT(0,0) = gamma(i,0);
    Phi_TT(0,0) = Phi(i,0);
    A_TT.row(0) = A.row(i);
    tau_TT.row(0) = tau.row(i);

    nu_TT.slice(1) = nu.slice(i);
    chi_TT.slice(1) = chi.slice(i);
    pi_TT.col(1) = pi.col(i);
    sigma_TT(1) = sigma(i);
    Z_TT.slice(1) = Z.slice(i);
    delta_TT.col(1) = delta.col(i);
    gamma_TT(1,0) = gamma(i,0);
    Phi_TT(1,0) = Phi(i,0);
    A_TT.row(1) = A.row(i);
    tau_TT.row(1) = tau.row(i);
    for(int j = 0; j < n_funct; j++){
      y_star_TT(j,0).row(0) = y_star(j,0).row(0);
    }

    temp_ind = 0;

    // Perform tempered transitions
    for(int l = 1; l < ((2 * N_t) + 1); l++){
      updateZTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                      Phi_TT(l,0), nu_TT.slice(l), chi_TT.slice(l),
                      pi_TT.col(l), sigma_TT(l), rho, l, (2 * N_t) + 1, Z_ph,
                      Z_TT);
      updatePi(alpha_3, Z_TT.slice(l), l, (2 * N_t) + 1,  pi_TT);

      tilde_tau(0) = delta_TT(0, l);
      for(int j = 1; j < M; j++){
        tilde_tau(j) = tilde_tau(j-1) * delta_TT(j,l);
      }

      updatePhiTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                        nu_TT.slice(l), gamma_TT(l,0), tilde_tau, Z_TT.slice(l),
                        chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, m_1, M_1,
                        Phi_TT);
      updateDelta(Phi_TT(l,0), gamma_TT(l,0), A_TT.row(l).t(), l, (2 * N_t) + 1,
                  delta_TT);

      updateA(alpha1l, beta1l, alpha2l, beta2l, delta_TT.col(l), var_epsilon1,
              var_epsilon2, l, (2 * N_t) + 1, A_TT);
      updateGamma(nu_1, delta_TT.col(l), Phi_TT(l,0), l, (2 * N_t) + 1,
                  gamma_TT);
      updateNuTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                       tau_TT.row(l).t(), Phi_TT(l,0), Z_TT.slice(l),
                       chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, P_mat,
                       b_1, B_1, nu_TT);
      updateTau(alpha, beta, nu_TT.slice(l), l, (2 * N_t) + 1, P_mat, tau_TT);
      updateSigmaTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                          alpha_0, beta_0, nu_TT.slice(l), Phi_TT(l,0),
                          Z_TT.slice(l), chi_TT.slice(l), l, (2 * N_t) + 1,
                          sigma_TT);
      updateChiTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                        Phi_TT(l,0), nu_TT.slice(l), Z_TT.slice(l), sigma_TT(l),
                        l, (2 * N_t) + 1, chi_TT);
      updateYStarTempered(beta_ladder(temp_ind), B_star, nu_TT.slice(l),
                          Phi_TT(l,0), Z_TT.slice(l), chi_TT.slice(l),
                          sigma_TT(l), l, (2 * N_t) + 1, y_star_TT);

      // update temp_ind
      if(l < N_t){
        temp_ind = temp_ind + 1;
      }
      if(l > N_t){
        temp_ind = temp_ind - 1;
      }
    }


    // get probability of acceptance
    logA = CalculateTTAcceptance(beta_ladder, y_obs, y_star_TT, B_obs, B_star,
                                 nu_TT, Phi_TT, Z_TT, chi_TT, sigma_TT);
    logu = std::log(R::runif(0,1));

    Rcpp::Rcout << "prob_accept: " << logA<< "\n";
    Rcpp::Rcout << "logu: " << logu<< "\n";
    // Accept or reject new state
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
      for(int j = 0; j < n_funct; j++){
        y_star(j,0).row(i % r_stored_iters) = y_star_TT(j,0).row(2 * N_t);
      }

      //update accept number
      accept_num = accept_num + 1;
    }

    //initialize next state
    if(((i+1) % r_stored_iters) != 0)
    nu.slice((i+1) % r_stored_iters) = nu.slice(i % r_stored_iters);
    chi.slice((i+1) % r_stored_iters) = chi.slice(i % r_stored_iters);
    pi.col((i+1) % r_stored_iters) = pi.col(i % r_stored_iters);
    sigma((i+1) % r_stored_iters) = sigma(i % r_stored_iters);
    Z.slice((i+1) % r_stored_iters) = Z.slice(i % r_stored_iters);
    delta.col((i+1) % r_stored_iters) = delta.col(i % r_stored_iters);
    A.row((i+1) % r_stored_iters) = A.row(i % r_stored_iters);
    tau.row((i+1) % r_stored_iters) = tau.row(i % r_stored_iters);
    Phi((i+1) % r_stored_iters,0) = Phi(i % r_stored_iters, 0);
    for(int j = 0; j < n_funct; j++){
      y_star(j,0).row((i+1) % r_stored_iters) = y_star(j,0).row(i % r_stored_iters);
    }

    // Calculate log likelihood
    loglik((i % r_stored_iters)) =  calcLikelihood(y_obs, y_star, B_obs, B_star, nu.slice((i % r_stored_iters)),
           Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),(i % r_stored_iters), sigma((i % r_stored_iters)));
    if(((i+1) % 5) == 0){
      Rcpp::Rcout << "Iteration: " << i+1 << "\n";
      Rcpp::Rcout << "Accpetance Probability: " << accept_num / (i+1.0) << "\n";
      Rcpp::Rcout << "Log-likelihood: " << arma::mean(loglik.subvec((i % r_stored_iters)-4, (i % r_stored_iters))) << "\n";
      Rcpp::checkUserInterrupt();
    }
    if(((i+1) % r_stored_iters) == 0 && i > 1){
      // Save parameters
      arma::cube nu1(K, P, r_stored_iters, arma::fill::randn);
      arma::cube chi1(n_funct, M, r_stored_iters, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters,1);
      arma::field<arma::cube> Phi1(r_stored_iters, 1);
      arma::mat tau1(r_stored_iters, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      arma::field<arma::mat> y_star1(n_funct, 1);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);
      for(int j = 0; j < n_funct; j++){
        y_star1(j,0) = arma::randn(r_stored_iters, t_star(j,0).n_elem);
        y_star1(j,0).row(0) = y_star(j,0).row(0);
      }

      for(int p=1; p < r_stored_iters; p++){
        nu1.slice(p) = nu.slice(p);
        chi1.slice(p) = chi.slice(p );
        pi1.col(p) = pi.col(p);
        sigma1(p) = sigma(p);
        A1.row(p) = A.row(p);
        Z1.slice(p) = Z.slice(p);
        delta1.col(p) = delta.col(p);
        gamma1(p,0) = gamma(p,0);
        Phi1(p,0) = Phi(p,0);
        tau1.row(p) = tau.row(p);
        for(int j = 0; j < n_funct; j++){
          y_star1(j,0).row(p) = y_star(j,0).row(p);
        }
      }

      nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
      y_star1.save(directory + "Y_Star" + std::to_string(q) + ".txt");
      chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
      pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
      A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
      delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
      sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
      tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
      gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
      Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
      Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

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
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
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
  return params;
}


//' Conducts tempered MCMC to estimate the posterior distribution in an unsupervised setting. MCMC samples will be stored in batches to a specified path.
//'
//' @name BFOC_U_TT
//' @param y_obs Field (list) of vectors containing the observed values
//' @param t_obs Field (list) of vectors containing time points of observed values
//' @param n_funct Double containing number of functions observed
//' @param P Int that indicates the number of b-spline basis functions
//' @param M int that indicates the number of slices used in Phi parameter
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param r_stored_iters Int constaining number of iterations performed for each batch
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
//' @param directory String containing path to store batches of MCMC samples
//' @param Z_est Matrix containing initial starting point of Z matrix
//' @param A_est Vector containing initial starting point of A vector
//' @param pi_est Vector containing initial starting point of pi vector
//' @param tau_est Vector containing initial starting point of tau vector
//' @param delta_est Vector containing initial starting point of delta vector
//' @param nu_est Matrix containing initial starting point of nu matrix
//' @param Phi_est Cube containing initial starting point of Phi matrix
//' @param gamma_est Cube containing initial starting point of gamma matrix
//' @param chi_est Matrix containing initial starting point of chi matrix
//' @param y_star_est Field of Vectors containing initial starting point of y_star
//' @param sigma_est Double containing starting point of sigma parameter
//' @returns params List of objects containing the MCMC samples from the last batch
//' @export
// [[Rcpp::export]]
Rcpp::List BFOC_U_Temp(const arma::field<arma::vec>& y_obs,
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
                       const std::string directory,
                       const arma::mat& Z_est,
                       const arma::vec& A_est,
                       const arma::vec& pi_est,
                       const arma::vec& tau_est,
                       const arma::vec& delta_est,
                       const arma::mat& nu_est,
                       const arma::cube& Phi_est,
                       const arma::cube& gamma_est,
                       const arma::mat& chi_est,
                       const arma::field<arma::vec>& y_star_est,
                       const double& temp,
                       const double& sigma_est){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  arma::field<arma::mat> B_star(n_funct,1);
  arma::field<arma::mat> y_star(n_funct, 1);
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
      y_star(i,0).row(0) = y_star_est(i,0).t();
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
  nu.slice(0) = nu_est;
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  chi.slice(0) = chi_est;
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  pi.col(0) = pi_est;
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  sigma(0) = sigma_est;
  arma::mat Z_ph(n_funct, K, arma::fill::ones);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));

  Z.slice(0) = Z_est;
  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  delta.col(0) = delta_est;
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  A.row(0) = A_est.t();
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }
  gamma(0,0) = gamma_est;
  Phi(0,0) = Phi_est;

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);
  tau.row(0) = tau_est.t();

  arma::vec b_1(P, arma::fill::zeros);
  arma::mat B_1(P, P, arma::fill::zeros);

  for(int i = 0; i < tot_mcmc_iters; i++){
    updateZTempered(temp, y_obs, y_star, B_obs, B_star, Phi((i % r_stored_iters),0), nu.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
            pi.col((i % r_stored_iters)), sigma((i % r_stored_iters)), rho, (i % r_stored_iters), r_stored_iters, Z_ph, Z);
    updatePi(alpha_3, Z.slice((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters,  pi);

    tilde_tau(0) = delta(0, (i % r_stored_iters));
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta(j,(i % r_stored_iters));
    }
    updatePhiTempered(temp, y_obs, y_star, B_obs, B_star, nu.slice((i % r_stored_iters)), gamma((i % r_stored_iters),0),
              tilde_tau, Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)), (i % r_stored_iters),
              r_stored_iters, m_1, M_1, Phi);
    updateDelta(Phi((i % r_stored_iters),0), gamma((i % r_stored_iters),0), A.row(i % r_stored_iters).t(), (i % r_stored_iters), r_stored_iters,
                delta);
    updateA(alpha1l, beta1l, alpha2l, beta2l, delta.col((i % r_stored_iters)), var_epsilon1,
            var_epsilon2, (i % r_stored_iters), r_stored_iters, A);
    updateGamma(nu_1, delta.col((i % r_stored_iters)), Phi((i % r_stored_iters),0), (i % r_stored_iters), r_stored_iters, gamma);
    updateNuTempered(temp, y_obs, y_star, B_obs, B_star, tau.row((i % r_stored_iters)).t(), Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)),
             chi.slice((i % r_stored_iters)), sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, P_mat, b_1, B_1, nu);
    updateTau(alpha, beta, nu.slice((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, P_mat, tau);
    updateSigmaTempered(temp, y_obs, y_star, B_obs, B_star, alpha_0, beta_0, nu.slice((i % r_stored_iters)),
                Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, sigma);
    updateChiTempered(temp, y_obs, y_star, B_obs, B_star, Phi((i % r_stored_iters),0), nu.slice((i % r_stored_iters)), Z.slice((i % r_stored_iters)),
              sigma((i % r_stored_iters)), (i % r_stored_iters), r_stored_iters, chi);
    updateYStarTempered(temp, B_star, nu.slice((i % r_stored_iters)), Phi((i % r_stored_iters),0), Z.slice((i % r_stored_iters)), chi.slice((i % r_stored_iters)),
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
      arma::cube nu1(K, P, r_stored_iters/50, arma::fill::randn);
      arma::cube chi1(n_funct, M, r_stored_iters/50, arma::fill::randn);
      arma::mat pi1(K, r_stored_iters/50, arma::fill::zeros);
      arma::vec sigma1(r_stored_iters/50, arma::fill::ones);
      arma::mat A1 = arma::ones(r_stored_iters/50, 2);
      arma::cube Z1 = arma::randi<arma::cube>(n_funct, K, r_stored_iters/50,
                                              arma::distr_param(0,1));
      arma::mat delta1(M, r_stored_iters/50, arma::fill::ones);
      arma::field<arma::cube> gamma1(r_stored_iters/50,1);
      arma::field<arma::cube> Phi1(r_stored_iters/50, 1);
      arma::mat tau1(r_stored_iters/50, K, arma::fill::ones);

      nu1.slice(0) = nu.slice(0);
      chi1.slice(0) = chi.slice(0);
      pi1.col(0) = pi.col(0);
      sigma1(0) = sigma(0);
      A1.row(0) = A.row(0);
      Z1.slice(0) = Z.slice(0);
      delta1.col(0) = delta.col(0);
      arma::field<arma::mat> y_star1(n_funct, 1);
      gamma1(0,0) = gamma(0,0);
      Phi1(0,0) = Phi(0,0);
      tau1.row(0) = tau.row(0);
      for(int j = 0; j < n_funct; j++){
        y_star1(j,0) = arma::randn(r_stored_iters/50, t_star(j,0).n_elem);
        y_star1(j,0).row(0) = y_star(j,0).row(0);
      }

      for(int p=1; p < r_stored_iters / 50; p++){
        nu1.slice(p) = nu.slice(50*p - 1);
        chi1.slice(p) = chi.slice(50*p - 1);
        pi1.col(p) = pi.col(50*p - 1);
        sigma1(p) = sigma(50*p - 1);
        A1.row(p) = A.row(50*p - 1);
        Z1.slice(p) = Z.slice(50*p - 1);
        delta1.col(p) = delta.col(p);
        gamma1(p,0) = gamma(50*p - 1,0);
        Phi1(p,0) = Phi(50*p  - 1,0);
        tau1.row(p) = tau.row(50*p - 1);
        for(int j = 0; j < n_funct; j++){
          y_star1(j,0).row(p) = y_star(j,0).row(50*p - 1);
        }
      }

      nu1.save(directory + "Nu" + std::to_string(q) + ".txt", arma::arma_ascii);
      y_star1.save(directory + "Y_Star" + std::to_string(q) + ".txt");
      chi1.save(directory + "Chi" + std::to_string(q) +".txt", arma::arma_ascii);
      pi1.save(directory + "Pi" + std::to_string(q) +".txt", arma::arma_ascii);
      A1.save(directory + "A" + std::to_string(q) +".txt", arma::arma_ascii);
      delta1.save(directory + "Delta" + std::to_string(q) +".txt", arma::arma_ascii);
      sigma1.save(directory + "Sigma" + std::to_string(q) +".txt", arma::arma_ascii);
      tau1.save(directory + "Tau" + std::to_string(q) +".txt", arma::arma_ascii);
      gamma1.save(directory + "Gamma" + std::to_string(q) +".txt");
      Phi1.save(directory + "Phi" + std::to_string(q) +".txt");
      Z1.save(directory + "Z" + std::to_string(q) +".txt", arma::arma_ascii);

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
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu),
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
  return params;
}

//' Conducts tempered MCMC to estimate the posterior distribution in an unsupervised setting. MCMC samples will be stored in batches to a specified path.
//'
//' @name BFOC_U_TT
//' @param y_obs Field (list) of vectors containing the observed values
//' @param t_obs Field (list) of vectors containing time points of observed values
//' @param n_funct Double containing number of functions observed
//' @param P Int that indicates the number of b-spline basis functions
//' @param M int that indicates the number of slices used in Phi parameter
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param r_stored_iters Int constaining number of iterations performed for each batch
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
//' @param directory String containing path to store batches of MCMC samples
//' @param Z_est Matrix containing initial starting point of Z matrix
//' @param A_est Vector containing initial starting point of A vector
//' @param pi_est Vector containing initial starting point of pi vector
//' @param tau_est Vector containing initial starting point of tau vector
//' @param delta_est Vector containing initial starting point of delta vector
//' @param nu_est Matrix containing initial starting point of nu matrix
//' @param Phi_est Cube containing initial starting point of Phi matrix
//' @param gamma_est Cube containing initial starting point of gamma matrix
//' @param chi_est Matrix containing initial starting point of chi matrix
//' @param y_star_est Field of Vectors containing initial starting point of y_star
//' @param sigma_est Double containing starting point of sigma parameter
//' @returns params List of objects containing the MCMC samples from the last batch
//' @export
// [[Rcpp::export]]
Rcpp::List BFOC_U_Templadder(const arma::field<arma::vec>& y_obs,
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
                             const arma::mat& Z_est,
                             const arma::vec& A_est,
                             const arma::vec& pi_est,
                             const arma::vec& tau_est,
                             const arma::vec& delta_est,
                             const arma::mat& nu_est,
                             const arma::cube& Phi_est,
                             const arma::cube& gamma_est,
                             const arma::mat& chi_est,
                             const arma::field<arma::vec>& y_star_est,
                             const double& beta_N_t,
                             const int& N_t,
                             const double& sigma_est){
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  arma::field<arma::mat> B_star(n_funct,1);
  arma::field<arma::mat> y_star(n_funct, 1);
  arma::field<arma::mat> y_star_TT(n_funct, 1);
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

      // create placeholder for tempered transitions
      y_star_TT(i,0) = arma::randn((2 * N_t) + 1, t_star(i,0).n_elem);
      y_star(i,0).row(0) = y_star_est(i,0).t();
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
  nu.slice(0) = nu_est;
  arma::cube chi(n_funct, M, r_stored_iters, arma::fill::randn);
  chi.slice(0) = chi_est;
  arma::mat pi(K, r_stored_iters, arma::fill::zeros);
  pi.col(0) = pi_est;
  arma::vec sigma(r_stored_iters, arma::fill::ones);
  sigma(0) = sigma_est;
  arma::mat Z_ph(n_funct, K, arma::fill::ones);
  arma::cube Z = arma::randi<arma::cube>(n_funct, K, r_stored_iters,
                                         arma::distr_param(0,1));
  Z.slice(0) = Z_est;
  arma::mat delta(M, r_stored_iters, arma::fill::ones);
  delta.col(0) = delta_est;
  arma::field<arma::cube> gamma(r_stored_iters,1);
  arma::field<arma::cube> Phi(r_stored_iters, 1);
  arma::vec tilde_tau(M, arma::fill::ones);
  arma::mat A = arma::ones(r_stored_iters, 2);
  A.row(0) = A_est.t();
  arma::vec loglik = arma::zeros(r_stored_iters);

  // start numbering for output files
  int q = 0;

  for(int i = 0; i < r_stored_iters; i++){
    gamma(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi(i,0) = arma::randn(K, P, M);
  }
  gamma(0,0) = gamma_est;
  Phi(0,0) = Phi_est;

  arma::vec m_1(P, arma::fill::zeros);
  arma::mat M_1(P, P, arma::fill::zeros);
  arma::mat tau(r_stored_iters, K, arma::fill::ones);
  tau.row(0) = tau_est.t();

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

  for(int i = 0; i < ((2 * N_t) + 1); i++){
    gamma_TT(i,0) = arma::cube(K, P, M, arma::fill::ones);
    Phi_TT(i,0) = arma::randn(K, P, M);
  }

  arma::mat tau_TT((2 * N_t) + 1, K, arma::fill::ones);

  int temp_ind = 0;
  double logA = 0;
  double logu = 0;
  int accept_num = 0;


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
  for(int j = 0; j < n_funct; j++){
    y_star_TT(j,0).row(0) = y_star(j,0).row(0);
  }

  temp_ind = 0;

  // Perform tempered transitions
  for(int l = 1; l < ((2 * N_t) + 1); l++){
    updateZTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                    Phi_TT(l,0), nu_TT.slice(l), chi_TT.slice(l),
                    pi_TT.col(l), sigma_TT(l), rho, l, (2 * N_t) + 1, Z_ph,
                    Z_TT);
    updatePi(alpha_3, Z_TT.slice(l), l, (2 * N_t) + 1,  pi_TT);

    tilde_tau(0) = delta_TT(0, l);
    for(int j = 1; j < M; j++){
      tilde_tau(j) = tilde_tau(j-1) * delta_TT(j,l);
    }

    updatePhiTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                      nu_TT.slice(l), gamma_TT(l,0), tilde_tau, Z_TT.slice(l),
                      chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, m_1, M_1,
                      Phi_TT);
    updateDelta(Phi_TT(l,0), gamma_TT(l,0), A_TT.row(l).t(), l, (2 * N_t) + 1,
                delta_TT);

    updateA(alpha1l, beta1l, alpha2l, beta2l, delta_TT.col(l), var_epsilon1,
            var_epsilon2, l, (2 * N_t) + 1, A_TT);
    updateGamma(nu_1, delta_TT.col(l), Phi_TT(l,0), l, (2 * N_t) + 1,
                gamma_TT);
    updateNuTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                     tau_TT.row(l).t(), Phi_TT(l,0), Z_TT.slice(l),
                     chi_TT.slice(l), sigma_TT(l), l, (2 * N_t) + 1, P_mat,
                     b_1, B_1, nu_TT);
    updateTau(alpha, beta, nu_TT.slice(l), l, (2 * N_t) + 1, P_mat, tau_TT);
    updateSigmaTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                        alpha_0, beta_0, nu_TT.slice(l), Phi_TT(l,0),
                        Z_TT.slice(l), chi_TT.slice(l), l, (2 * N_t) + 1,
                        sigma_TT);
    updateChiTempered(beta_ladder(temp_ind), y_obs, y_star_TT, B_obs, B_star,
                      Phi_TT(l,0), nu_TT.slice(l), Z_TT.slice(l), sigma_TT(l),
                      l, (2 * N_t) + 1, chi_TT);
    updateYStarTempered(beta_ladder(temp_ind), B_star, nu_TT.slice(l),
                        Phi_TT(l,0), Z_TT.slice(l), chi_TT.slice(l),
                        sigma_TT(l), l, (2 * N_t) + 1, y_star_TT);

    // update temp_ind
    if(l < N_t){
      temp_ind = temp_ind + 1;
    }
    if(l > N_t){
      temp_ind = temp_ind - 1;
    }


  }
  Rcpp::List params = Rcpp::List::create(Rcpp::Named("nu", nu_TT),
                                         Rcpp::Named("y_star", y_star_TT),
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
