#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFPMM.h>

// Tests updating Z using partial membership model
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
    Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
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
  arma::vec pi = {1, 1, 1};

  // Initialize placeholder
  arma::vec Z_ph = arma::zeros(3);


  //Initialize Z_samp
  arma::cube Z_samp = arma::ones(20, 3, 500);
  for(int i = 0; i < 20; i++){
    Z_samp.slice(0).row(i) = BayesFPMM::rdirichlet(pi).t();
  }
  for(int i = 0; i < 500; i++){
    BayesFPMM::updateZ_PM(y_obs, B_obs, Phi, nu, chi, pi,
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

  arma::cube mod = arma::zeros(20, 3, 2);
  mod.slice(0) = Z_est;
  mod.slice(1) = Z;
  return mod;
}

// Tests updating Z using partial membership model
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
    Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
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
  arma::vec pi = {1, 1, 1};

  // Initialize placeholder
  arma::vec Z_ph = arma::zeros(3);


  //Initialize Z_samp
  arma::cube Z_samp = arma::ones(20, 3, 500);
  for(int i = 0; i < 20; i++){
    Z_samp.slice(0).row(i) = BayesFPMM::rdirichlet(pi).t();
  }
  double beta = 0.05;

  for(int i = 0; i < 500; i++){
    BayesFPMM::updateZTempered_PM(beta, y_obs, B_obs, Phi, nu, chi, pi,
               sigma_sq, i, 500, 1.0, 2000, Z_ph, Z_samp);
  }

  arma::mat Z_est = arma::zeros(20, 3);
  arma::vec ph_Z = arma::zeros(500);
  for(int i = 0; i < 20; i++){
    for(int j = 0; j < 3; j++){
      for(int l = 200; l < 500; l++){
        ph_Z(l - 200) = Z_samp(i,j,l);
      }
      Z_est(i,j) = arma::median(ph_Z);
    }
  }

  arma::cube mod = arma::zeros(20, 3, 2);
  mod.slice(0) = Z_est;
  mod.slice(1) = Z;
  return mod;
}

// Tests sampling of Z
context("Functional model Z unit test") {
  test_that("Sampler for Z is working") {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateZ_PM();
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

  test_that("Sampler for Z using tempered transitions is working") {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::cube x = TestUpdateTemperedZ_PM();
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
}
