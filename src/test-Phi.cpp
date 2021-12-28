#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFPMM.h>

//' Tests updating Phi
//'
arma::field<arma::cube> TestUpdatePhi(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat{bspline.basis(true)};
  // Make B_obs
  arma::field<arma::mat> B_obs(40,1);


  for(int i = 0; i < 40; i++)
  {
    B_obs(i,0) = bspline_mat;
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,2);
  for(int i=0; i < 2; i++)
  {
    Phi.slice(i) = (2-i) * arma::randn<arma::mat>(3,8);
  }
  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(40, 2, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(40, 3);
  arma::mat alpha(40,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::field<arma::vec> y_obs(40, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  // Initialize pi
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::vec tilde_tau = {1, 2};
  for(int i = 0; i < 250; i++){
    BayesFPMM::updatePhi(y_obs, B_obs, nu, gamma, tilde_tau, Z, chi,
                         sigma_sq, i, 250, m_1, M_1, Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

//' Tests updating Phi
//'
arma::field<arma::cube> TestUpdatePhiTempered(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat{bspline.basis(true)};
  // Make B_obs
  arma::field<arma::mat> B_obs(40,1);


  for(int i = 0; i < 40; i++)
  {
    B_obs(i,0) = bspline_mat;
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,2);
  for(int i=0; i < 2; i++)
  {
    Phi.slice(i) = (2-i) * arma::randn<arma::mat>(3,8);
  }
  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(40, 2, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(40, 3);
  arma::mat alpha(40,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::field<arma::vec> y_obs(40, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  // Initialize pi
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::vec tilde_tau = {1, 2};
  for(int i = 0; i < 250; i++){
    BayesFPMM::updatePhiTempered(0.5, y_obs, B_obs, nu, gamma, tilde_tau, Z, chi,
                                 sigma_sq, i, 250, m_1, M_1, Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

context("Unit tests for Phi parameters") {
  test_that("Sampler for Phi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhi();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Tempered sampler for Phi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhiTempered();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }
  //
  // test_that("Sampler for Nu parameters in multivariate model"){
  //   Rcpp::Environment base_env("package:base");
  //   Rcpp::Function set_seed_r = base_env["set.seed"];
  //   set_seed_r(1);
  //   arma::cube x = TestUpdateNuMV();
  //   arma::mat est = x.slice(0);
  //   arma::mat truth = x.slice(1);
  //   bool similar = true;
  //   for(int i = 0; i < est.n_rows; i++){
  //     for(int j = 0; j < est.n_cols; j++){
  //       if(std::abs(est(i,j) - truth(i,j)) > 0.3){
  //         similar = false;
  //       }
  //     }
  //   }
  //   expect_true(similar == true);
  // }
  //
  // test_that("Tempered sampler for Nu parameters in multivariate model"){
  //   Rcpp::Environment base_env("package:base");
  //   Rcpp::Function set_seed_r = base_env["set.seed"];
  //   set_seed_r(1);
  //   arma::cube x = TestUpdateNuTempered();
  //   arma::mat est = x.slice(0);
  //   arma::mat truth = x.slice(1);
  //   bool similar = true;
  //   for(int i = 0; i < est.n_rows; i++){
  //     for(int j = 0; j < est.n_cols; j++){
  //       if(std::abs(est(i,j) - truth(i,j)) > 0.6){
  //         similar = false;
  //       }
  //     }
  //   }
  //   expect_true(similar == true);
  // }
  //
  // test_that("Sampler for Tau parameters"){
  //   Rcpp::Environment base_env("package:base");
  //   Rcpp::Function set_seed_r = base_env["set.seed"];
  //   set_seed_r(1);
  //   arma::mat x = TestUpdateTau();
  //   arma::vec est = x.row(0).t();
  //   arma::vec truth = x.row(1).t();
  //   bool similar = true;
  //   for(int i = 0; i < est.n_elem; i++){
  //     if(std::abs(est(i) - truth(i)) > 0.5){
  //       similar = false;
  //     }
  //   }
  //   expect_true(similar == true);
  // }
  //
  // test_that("Sampler for Tau parameters for multivariate model"){
  //   Rcpp::Environment base_env("package:base");
  //   Rcpp::Function set_seed_r = base_env["set.seed"];
  //   set_seed_r(1);
  //   arma::mat x = TestUpdateTauMV();
  //   arma::vec est = x.row(0).t();
  //   arma::vec truth = x.row(1).t();
  //   bool similar = true;
  //   for(int i = 0; i < est.n_elem; i++){
  //     if(std::abs(est(i) - truth(i)) > 0.5){
  //       similar = false;
  //     }
  //   }
  //   expect_true(similar == true);
  // }
}

