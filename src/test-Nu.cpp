#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFMMM.h>

// Tests updating Nu
//
// @name TestUpdateNu
arma::cube TestUpdateNu(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (20 x 8)
  arma::mat bspline_mat{bspline.basis(true)};
  // Make B_obs
  arma::field<arma::mat> B_obs(20,1);

  for(int i = 0; i < 20; i++)
  {
    B_obs(i,0) = bspline_mat;
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,5);
  for(int i=0; i < 5; i++)
  {
    Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.01;

  // Make chi matrix
  arma::mat chi(20, 5, arma::fill::randn);


  //Make Z
  arma::mat Z(20, 3);
  arma::vec c(3, arma::fill::ones);
  arma::vec pi = BayesFMMM::rdirichlet(c);

  // setting alpha_3 = 10
  arma:: vec alpha = pi * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha).t();
  }

  arma::field<arma::vec> y_obs(20, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 20; j++){
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

  arma::cube Nu_samp = arma::randn(nu.n_rows, nu.n_cols, 500);
  arma::vec b_1(nu.n_cols, arma::fill::zeros);
  arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
  arma::mat P(nu.n_cols, nu.n_cols, arma::fill::zeros);
  P.zeros();
  for(int j = 0; j < P.n_rows; j++){
    P(0,0) = 1;
    if(j > 0){
      P(j,j) = 2;
      P(j-1,j) = -1;
      P(j,j-1) = -1;
    }
    P(P.n_rows - 1, P.n_rows - 1) = 1;
  }
  arma::vec tau(nu.n_rows, arma::fill::ones);
  tau = tau / 10;
  for(int i = 0; i < 500; i++){
    BayesFMMM::updateNu(y_obs, B_obs, tau, Phi, Z, chi, sigma_sq, i, 500,
             P, b_1, B_1, Nu_samp);
  }

  arma::cube mod = arma::zeros(3,8,2);
  arma::vec nu_ph = arma::zeros(200);
  arma::mat nu_est = arma::zeros(3,8);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int k = 300; k < 500; k++){
        nu_ph(k - 300) = Nu_samp(i,j,k);
      }
      nu_est(i,j) = arma::median(nu_ph);
    }
  }
  mod.slice(1) = nu;
  mod.slice(0) = nu_est;
  return mod;
}

// Tests updating Nu
//
// @name TestUpdateNu
arma::cube TestUpdateNuTempered(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (20 x 8)
  arma::mat bspline_mat{bspline.basis(true)};
  // Make B_obs
  arma::field<arma::mat> B_obs(20,1);

  for(int i = 0; i < 20; i++)
  {
    B_obs(i,0) = bspline_mat;
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,5);
  for(int i=0; i < 5; i++)
  {
    Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.01;

  // Make chi matrix
  arma::mat chi(20, 5, arma::fill::randn);


  //Make Z
  arma::mat Z(20, 3);
  arma::vec c(3, arma::fill::ones);
  arma::vec pi = BayesFMMM::rdirichlet(c);

  // setting alpha_3 = 10
  arma:: vec alpha = pi * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha).t();
  }

  arma::field<arma::vec> y_obs(20, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 20; j++){
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

  arma::cube Nu_samp = arma::randn(nu.n_rows, nu.n_cols, 500);
  arma::vec b_1(nu.n_cols, arma::fill::zeros);
  arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
  arma::mat P(nu.n_cols, nu.n_cols, arma::fill::zeros);
  P.zeros();
  for(int j = 0; j < P.n_rows; j++){
    P(0,0) = 1;
    if(j > 0){
      P(j,j) = 2;
      P(j-1,j) = -1;
      P(j,j-1) = -1;
    }
    P(P.n_rows - 1, P.n_rows - 1) = 1;
  }
  arma::vec tau(nu.n_rows, arma::fill::ones);
  tau = tau / 10;
  for(int i = 0; i < 500; i++){
    BayesFMMM::updateNuTempered(0.6, y_obs, B_obs, tau, Phi, Z, chi, sigma_sq, i, 500,
                        P, b_1, B_1, Nu_samp);
  }

  arma::cube mod = arma::zeros(3,8,2);
  arma::vec nu_ph = arma::zeros(200);
  arma::mat nu_est = arma::zeros(3,8);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int k = 300; k < 500; k++){
        nu_ph(k - 300) = Nu_samp(i,j,k);
      }
      nu_est(i,j) = arma::median(nu_ph);
    }
  }
  mod.slice(1) = nu;
  mod.slice(0) = nu_est;
  return mod;
}

// Tests updating Nu
//
// @name TestUpdateNu
arma::cube TestUpdateNuMV(){
  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,5);
  for(int i=0; i < 5; i++)
  {
    Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(20, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(20, 3);
  arma::mat alpha(20,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::mat y_obs = arma::zeros(20, 8);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 20; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs.row(j) = arma::mvnrnd(mean, sigma_sq *
      arma::eye(mean.n_elem, mean.n_elem)).t();
  }

  arma::cube Nu_samp = arma::randn(nu.n_rows, nu.n_cols, 500);
  arma::vec b_1(nu.n_cols, arma::fill::zeros);
  arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
  arma::vec tau(nu.n_rows, arma::fill::ones);
  tau = tau * 10;
  for(int i = 0; i < 500; i++){
    BayesFMMM::updateNuMV(y_obs, tau, Phi, Z, chi, sigma_sq, i, 500,
                          b_1, B_1, Nu_samp);
  }

  arma::cube mod = arma::zeros(3,8,2);
  arma::vec nu_ph = arma::zeros(200);
  arma::mat nu_est = arma::zeros(3,8);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int k = 300; k < 500; k++){
        nu_ph(k - 300) = Nu_samp(i,j,k);
      }
      nu_est(i,j) = arma::median(nu_ph);
    }
  }
  mod.slice(1) = nu;
  mod.slice(0) = nu_est;
  return mod;
}

// Tests updating Nu
//
// @name TestUpdateNu
arma::cube TestUpdateNuMVTempered(){
  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,5);
  for(int i=0; i < 5; i++)
  {
    Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(20, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(20, 3);
  arma::mat alpha(20,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::mat y_obs = arma::zeros(20, 8);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 20; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs.row(j) = arma::mvnrnd(mean, sigma_sq *
      arma::eye(mean.n_elem, mean.n_elem)).t();
  }

  arma::cube Nu_samp = arma::randn(nu.n_rows, nu.n_cols, 500);
  arma::vec b_1(nu.n_cols, arma::fill::zeros);
  arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
  arma::vec tau(nu.n_rows, arma::fill::ones);
  tau = tau * 10;
  for(int i = 0; i < 500; i++){
    BayesFMMM::updateNuTemperedMV(0.6, y_obs, tau, Phi, Z, chi, sigma_sq, i, 500,
                          b_1, B_1, Nu_samp);
  }

  arma::cube mod = arma::zeros(3,8,2);
  arma::vec nu_ph = arma::zeros(200);
  arma::mat nu_est = arma::zeros(3,8);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int k = 300; k < 500; k++){
        nu_ph(k - 300) = Nu_samp(i,j,k);
      }
      nu_est(i,j) = arma::median(nu_ph);
    }
  }
  mod.slice(1) = nu;
  mod.slice(0) = nu_est;
  return mod;
}

// Tests updating Tau
//
// @name TestUpdateTau
arma::mat TestUpdateTau(){
  arma::vec tau = {1, 2, 2, 3, 5, 6};
  arma::mat P(100, 100, arma::fill::zeros);
  P.zeros();
  for(int j = 0; j < P.n_rows; j++){
    P(0,0) = 1;
    if(j > 0){
      P(j,j) = 2;
      P(j-1,j) = -1;
      P(j,j-1) = -1;
    }
    P(P.n_rows - 1, P.n_rows - 1) = 1;
  }
  arma::mat nu(6, 100, arma::fill::zeros);
  arma::mat tau_samp(200, 6, arma::fill::zeros);
  arma::vec zeros_nu(100, arma::fill::zeros);
  for(int i = 0; i < 200; i++){
    for(int j = 0; j < 6; j++){
      nu.row(j) = arma::mvnrnd(zeros_nu, arma::pinv(P * tau(j))).t();
    }
    BayesFMMM::updateTau(1, 1, nu, i, 200, P, tau_samp);
  }
  arma::vec tau_est(6, arma::fill::zeros);
  for(int i = 0; i < 6; i++){
    tau_est(i) = arma::median(tau_samp.col(i));
  }
  arma::mat mod = arma::zeros(2, 6);
  mod.row(0) = tau_est.t();
  mod.row(1) = tau.t();
  return mod;
}

// Tests updating Tau
//
// @name TestUpdateTau
arma::mat TestUpdateTauMV(){
  arma::vec tau = {1, 2, 2, 3, 5, 6};
  arma::mat P(100, 100, arma::fill::zeros);
  P.zeros();
  for(int j = 0; j < P.n_rows; j++){
    P(j,j) = 1;
  }
  arma::mat nu(6, 100, arma::fill::zeros);
  arma::mat tau_samp(200, 6, arma::fill::zeros);
  arma::vec zeros_nu(100, arma::fill::zeros);
  for(int i = 0; i < 200; i++){
    for(int j = 0; j < 6; j++){
      nu.row(j) = arma::mvnrnd(zeros_nu, P * tau(j)).t();
    }
    BayesFMMM::updateTauMV(1, 1, nu, i, 200, tau_samp);
  }
  arma::vec tau_est(6, arma::fill::zeros);
  for(int i = 0; i < 6; i++){
    tau_est(i) = arma::median(tau_samp.col(i));
  }
  arma::mat mod = arma::zeros(2, 6);
  mod.row(0) = tau_est.t();
  mod.row(1) = tau.t();
  return mod;
}

context("Unit tests for Nu parameters") {
  test_that("Sampler for Nu parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateNu();
    arma::mat est = x.slice(0);
    arma::mat truth = x.slice(1);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.3){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Tempered sampler for Nu parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateNuTempered();
    arma::mat est = x.slice(0);
    arma::mat truth = x.slice(1);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.6){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Nu parameters in multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateNuMV();
    arma::mat est = x.slice(0);
    arma::mat truth = x.slice(1);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.3){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Tempered sampler for Nu parameters in multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateNuTempered();
    arma::mat est = x.slice(0);
    arma::mat truth = x.slice(1);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.6){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Tau parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::mat x = TestUpdateTau();
    arma::vec est = x.row(0).t();
    arma::vec truth = x.row(1).t();
    bool similar = true;
    for(int i = 0; i < est.n_elem; i++){
      if(std::abs(est(i) - truth(i)) > 0.5){
        similar = false;
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Tau parameters for multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::mat x = TestUpdateTauMV();
    arma::vec est = x.row(0).t();
    arma::vec truth = x.row(1).t();
    bool similar = true;
    for(int i = 0; i < est.n_elem; i++){
      if(std::abs(est(i) - truth(i)) > 0.5){
        similar = false;
      }
    }
    expect_true(similar == true);
  }
}

