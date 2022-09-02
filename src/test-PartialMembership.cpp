#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFMMM.h>

// Tests updating Z using mixed membership model
//
// @name TestUpdateZ_PM
arma::cube TestUpdateZ_PM(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat{bspline.basis(true)};
  // Make B_obs
  arma::field<arma::mat> B_obs(20,1);

  for(int i = 0; i < 20; i++){
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
    Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.0001;

  // Make chi matrix
  arma::mat chi(20, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(20, 3);
  arma::mat alpha(20,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
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

  // Initialize pi
  arma::vec pi = {10, 10, 10};

  // Initialize placeholder
  arma::vec Z_ph = arma::zeros(3);


  //Initialize Z_samp
  arma::cube Z_samp = arma::ones(20, 3, 500);
  for(int i = 0; i < 20; i++){
    Z_samp.slice(0).row(i) = BayesFMMM::rdirichlet(pi).t();
  }
  for(int i = 0; i < 500; i++){
    BayesFMMM::updateZ_PM(y_obs, B_obs, Phi, nu, chi, pi,
                          sigma_sq, i, 500, 1.0, 2000, Z_ph, Z_samp);
  }
  arma::mat Z_est = arma::zeros(20, 3);
  arma::vec ph_Z = arma::zeros(300);
  for(int i = 0; i < 20; i++){
    for(int j = 0; j < 3; j++){
      for(int l = 200; l < 500; l++){
        ph_Z(l - 200) = Z_samp(i,j,l);
      }
      Z_est(i,j) = arma::median(ph_Z);
    }
  }

  // normalize
  for(int i = 0; i < 20; i++){
    Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
  }

  arma::cube mod = arma::zeros(20, 3, 2);
  mod.slice(0) = Z_est;
  mod.slice(1) = Z;
  return mod;
}

// Tests updating Z using mixed membership model
//
// @name TestUpdateTemperedZ_PM
arma::cube TestUpdateTemperedZ_PM(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat{bspline.basis(true)};
  // Make B_obs
  arma::field<arma::mat> B_obs(20,1);

  for(int i = 0; i < 20; i++){
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
    Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.0001;

  // Make chi matrix
  arma::mat chi(20, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(20, 3);
  arma::mat alpha(20,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
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

  // Initialize pi
  arma::vec pi = {10, 10, 10};

  // Initialize placeholder
  arma::vec Z_ph = arma::zeros(3);


  //Initialize Z_samp
  arma::cube Z_samp = arma::ones(20, 3, 500);
  for(int i = 0; i < 20; i++){
    Z_samp.slice(0).row(i) = BayesFMMM::rdirichlet(pi).t();
  }
  double beta = 0.05;

  for(int i = 0; i < 500; i++){
    BayesFMMM::updateZTempered_PM(beta, y_obs, B_obs, Phi, nu, chi, pi,
               sigma_sq, i, 500, 1.0, 2000, Z_ph, Z_samp);
  }

  arma::mat Z_est = arma::zeros(20, 3);
  arma::vec ph_Z = arma::zeros(300);
  for(int i = 0; i < 20; i++){
    for(int j = 0; j < 3; j++){
      for(int l = 200; l < 500; l++){
        ph_Z(l - 200) = Z_samp(i,j,l);
      }
      Z_est(i,j) = arma::median(ph_Z);
    }
  }

  // normalize
  for(int i = 0; i < 20; i++){
    Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
  }
  arma::cube mod = arma::zeros(20, 3, 2);
  mod.slice(0) = Z_est;
  mod.slice(1) = Z;
  return mod;
}

// Tests updating Z in the multivariate scenario
//
// @name TestUpdateZ_MV
arma::cube TestUpdateZ_MV(){
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

  // Initialize pi
  arma::vec pi = {10, 10, 10};

  // Initialize placeholder
  arma::vec Z_ph = arma::zeros(3);

  //Initialize Z_samp
  arma::cube Z_samp = arma::ones(20, 3, 500);
  for(int i = 0; i < 20; i++){
    Z_samp.slice(0).row(i) = BayesFMMM::rdirichlet(pi).t();
  }
  for(int i = 0; i < 500; i++)
  {
    BayesFMMM::updateZ_MMMV(y_obs, Phi, nu, chi, pi,
                            sigma_sq, i, 500, 1.0, 2000, Z_ph, Z_samp);
  }
  arma::mat Z_est = arma::zeros(20, 3);
  arma::vec ph_Z = arma::zeros(300);
  for(int i = 0; i < 20; i++){
    for(int j = 0; j < 3; j++){
      for(int l = 200; l < 500; l++){
        ph_Z(l - 200) = Z_samp(i,j,l);
      }
      Z_est(i,j) = arma::median(ph_Z);
    }
  }

  // normalize
  for(int i = 0; i < 20; i++){
    Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
  }

  arma::cube mod = arma::zeros(20, 3, 2);
  mod.slice(0) = Z_est;
  mod.slice(1) = Z;
  return mod;
}


// Tests updating Z in the multivariate scenario usint tempered transitions
//
// @name TestUpdateTemperedZ_MV
arma::cube TestUpdateTemperedZ_MV(){
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

  // Initialize pi
  arma::vec pi = {10, 10, 10};

  // Initialize placeholder
  arma::vec Z_ph = arma::zeros(3);

  //Initialize Z_samp
  arma::cube Z_samp = arma::ones(20, 3, 500);
  for(int i = 0; i < 20; i++){
    Z_samp.slice(0).row(i) = BayesFMMM::rdirichlet(pi).t();
  }
  double beta = 0.05;

  for(int i = 0; i < 500; i++)
  {
    BayesFMMM::updateZTempered_MMMV(beta, y_obs, Phi, nu, chi, pi,
                                    sigma_sq, i, 500, 1.0, 2000, Z_ph, Z_samp);
  }
  arma::mat Z_est = arma::zeros(20, 3);
  arma::vec ph_Z = arma::zeros(300);
  for(int i = 0; i < 20; i++){
    for(int j = 0; j < 3; j++){
      for(int l = 200; l < 500; l++){
        ph_Z(l - 200) = Z_samp(i,j,l);
      }
      Z_est(i,j) = arma::median(ph_Z);
    }
  }
  // normalize
  for(int i = 0; i < 20; i++){
    Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
  }

  arma::cube mod = arma::zeros(20, 3, 2);
  mod.slice(0) = Z_est;
  mod.slice(1) = Z;
  return mod;
}

// Tests updating pi using mixed membership model
//
// @name TestUpdateZ_PM
arma::mat TestUpdatepi(){
  // Make Z matrix
  arma::mat Z(100, 3);
  arma::vec c(3, arma::fill::ones);
  arma::vec pi = BayesFMMM::rdirichlet(c);

  // setting alpha_3 = 100
  arma:: vec alpha = pi * 100;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha).t();
  }

  // Initialize placeholder
  arma::vec pi_ph = arma::zeros(3);


  //Initialize Z_samp
  arma::mat pi_samp = arma::ones(3, 500);

  pi_samp.col(0) = BayesFMMM::rdirichlet(c);

  for(int i = 0; i < 500; i++)
  {
    BayesFMMM::updatePi_PM(100, Z, c, i, 500, 1000, pi_ph, pi_samp);
  }

  arma::vec pi_est = arma::zeros(3);
  arma::vec ph_pi = arma::zeros(300);
  for(int j = 0; j < 3; j++){
    for(int l = 200; l < 500; l++){
      ph_pi(l - 200) = pi_samp(j,l);
    }
    pi_est(j) = arma::median(ph_pi);
  }
  // normalize
  for(int j = 0; j < 3; j++){
    pi_est = pi_est / arma::accu(pi_est);
  }

  arma::mat mod = arma::zeros(3, 2);
  mod.col(0) = pi_est;
  mod.col(1) = pi;
  return mod;
}

// Tests updating pi using mixed membership model
//
// @name TestUpdateZ_PM
double TestUpdatealpha3(){

  // Make Z matrix
  arma::mat Z(100, 3);
  arma::vec c(3, arma::fill::ones);
  arma::vec pi = BayesFMMM::rdirichlet(c);

  // setting alpha_3 = 10
  arma:: vec alpha = pi * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha).t();
  }

  arma::vec alpha_3(10000, arma::fill::ones);

  for(int i = 0; i < 10000; i++)
  {
    BayesFMMM::updateAlpha3(pi, 0.1, Z, i, 10000, 0.05, alpha_3);
  }
  double mod = arma::median(alpha_3.subvec(2000,9999));

  return mod;
}


// Tests sampling of Z
context("Unit tests for Z parameters") {
  test_that("Sampler for Z parameters") {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateZ_PM();
    arma::mat est = x.slice(0);
    arma::mat truth = x.slice(1);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.02){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Z using tempered transitions") {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateTemperedZ_PM();
    arma::mat est = x.slice(0);
    arma::mat truth = x.slice(1);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.02){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Z in multivariate case") {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateZ_MV();
    arma::mat est = x.slice(0);
    arma::mat truth = x.slice(1);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.02){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Z in multivariate case using tempered transitions") {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateTemperedZ_MV();
    arma::mat est = x.slice(0);
    arma::mat truth = x.slice(1);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.05){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for pi"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::mat x = TestUpdatepi();
    arma::vec est = x.col(0);
    arma::vec truth = x.col(1);
    bool similar = true;
    for(int i = 0; i < est.n_elem; i++){
      if(std::abs(est(i) - truth(i)) > 0.02){
        similar = false;
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for alpha_3"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    double x = TestUpdatealpha3();
    bool similar = true;
    if(std::abs(x - 10) > 2){
      similar = false;
    }
    expect_true(similar == true);
  }
}
