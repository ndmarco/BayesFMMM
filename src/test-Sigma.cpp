#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFPMM.h>

// Tests updating Sigma
//
arma::vec TestUpdateSigma(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat{bspline.basis(true)};
  // Make B_obs
  arma::field<arma::mat> B_obs(100,1);

  arma::field<arma::mat> B_star(100,1);


  for(int i = 0; i < 100; i++)
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
  double sigma_sq = 0.5;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  arma::mat Z(100, 3);
  arma::vec c(3, arma::fill::randu);

  // setting alpha_3 = 10
  arma:: vec alpha = c * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
  }

  arma::field<arma::vec> y_obs(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
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
  double alpha_0 = 1;
  double beta_0 = 1;
  arma::vec sigma_samp(1000, arma::fill::zeros);
  for(int i = 0; i < 1000; i++){
    BayesFPMM::updateSigma(y_obs, B_obs, alpha_0, beta_0, nu, Phi, Z, chi,
                           i, 1000, sigma_samp);
  }
  arma::vec mod = arma::zeros(2);
  mod(0) = arma::median(sigma_samp);
  mod(1) = sigma_sq;

  return mod;
}

// Tests updating Sigma using tempered transitions
//
arma::vec TestUpdateSigmaTempered(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat{bspline.basis(true)};
  // Make B_obs
  arma::field<arma::mat> B_obs(100,1);

  arma::field<arma::mat> B_star(100,1);


  for(int i = 0; i < 100; i++)
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
  double sigma_sq = 0.5;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  arma::mat Z(100, 3);
  arma::vec c(3, arma::fill::randu);

  // setting alpha_3 = 10
  arma:: vec alpha = c * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
  }

  arma::field<arma::vec> y_obs(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
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
  double alpha_0 = 1;
  double beta_0 = 1;
  arma::vec sigma_samp(1000, arma::fill::zeros);
  for(int i = 0; i < 1000; i++){
    BayesFPMM::updateSigmaTempered(0.5, y_obs, B_obs, alpha_0, beta_0, nu, Phi, Z, chi,
                           i, 1000, sigma_samp);
  }
  arma::vec mod = arma::zeros(2);
  mod(0) = arma::median(sigma_samp);
  mod(1) = sigma_sq;

  return mod;
}

// Tests updating Sigma for multivariate model
//
arma::vec TestUpdateSigmaMV(){
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
  double sigma_sq = 0.5;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  arma::mat Z(100, 3);
  arma::vec c(3, arma::fill::randu);

  // setting alpha_3 = 10
  arma:: vec alpha = c * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
  }

  arma::mat y_obs = arma::zeros(100, 8);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
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

  double alpha_0 = 1;
  double beta_0 = 1;
  arma::vec sigma_samp(1000, arma::fill::zeros);
  for(int i = 0; i < 1000; i++){
    BayesFPMM::updateSigmaMV(y_obs, alpha_0, beta_0, nu, Phi, Z, chi,
                             i, 1000, sigma_samp);
  }
  arma::vec mod = arma::zeros(2);
  mod(0) = arma::median(sigma_samp);
  mod(1) = sigma_sq;

  return mod;
}

// Tests updating Sigma for multivariate model
//
arma::vec TestUpdateSigmaTemperedMV(){
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
  double sigma_sq = 0.5;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  arma::mat Z(100, 3);
  arma::vec c(3, arma::fill::randu);

  // setting alpha_3 = 10
  arma:: vec alpha = c * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
  }

  arma::mat y_obs = arma::zeros(100, 8);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
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

  double alpha_0 = 1;
  double beta_0 = 1;
  arma::vec sigma_samp(1000, arma::fill::zeros);
  for(int i = 0; i < 1000; i++){
    BayesFPMM::updateSigmaTemperedMV(0.5, y_obs, alpha_0, beta_0, nu, Phi, Z, chi,
                                     i, 1000, sigma_samp);
  }
  arma::vec mod = arma::zeros(2);
  mod(0) = arma::median(sigma_samp);
  mod(1) = sigma_sq;

  return mod;
}

context("Unit tests for sigma parameter") {
  test_that("Sampler for sigma parameter"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::vec x = TestUpdateSigma();
    double est = x(0);
    double truth = x(1);
    bool similar = true;
    if(std::abs(est - truth) > 0.05){
      similar = false;
    }
    expect_true(similar == true);
  }

  test_that("Sampler for sigma parameter using tempered transitions"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::vec x = TestUpdateSigmaTempered();
    double est = x(0);
    double truth = x(1);
    bool similar = true;
    if(std::abs(est - truth) > 0.05){
      similar = false;
    }
    expect_true(similar == true);
  }

  test_that("Sampler for sigma parameter for the multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::vec x = TestUpdateSigmaMV();
    double est = x(0);
    double truth = x(1);
    bool similar = true;
    if(std::abs(est - truth) > 0.05){
      similar = false;
    }
    expect_true(similar == true);
  }
  test_that("Sampler for sigma parameter using tempered transitions for the multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::vec x = TestUpdateSigmaTemperedMV();
    double est = x(0);
    double truth = x(1);
    bool similar = true;
    if(std::abs(est - truth) > 0.05){
      similar = false;
    }
    expect_true(similar == true);
  }
}
