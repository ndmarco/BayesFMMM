#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFMMM.h>

// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdateEta(){
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

  // Make eta variables
  arma::cube eta(8, 2, 3);
  for(int i = 0; i < 3; i++){
    eta.slice(i) = arma::randn<arma::mat>(8,2);
  }

  // Make xi variables
  arma::field<arma::cube> xi(250,3);
  for(int i = 0; i < 3; i++){
    xi(0,i) = arma::zeros(8,2,2);
    for(int j =0; j < 2; j++){
      xi(0,i).slice(j) = (2-i) * 0.1 * arma::randn<arma::mat>(8,2);
    }
  }
  for(int i = 1; i < 250; i++){
    for(int j = 0; j < 3; j++){
      xi(i,j) = xi(0,j);
    }
  }

  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(40, 2, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(40, 3);
  arma::mat alpha(40,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::mat X = arma::randn<arma::mat>(40,2);
  arma::field<arma::vec> y_obs(40, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * (nu.row(l).t() + (eta.slice(l) * X.row(j).t()));
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * (Phi.slice(m).row(l).t() + (xi(0,l).slice(m)* X.row(j).t()));
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  // Initialize pi
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> eta_samp(250,1);

  for(int i = 0; i < 250; i++){
    eta_samp(i,0) = arma::zeros(8,2,3);
  }

  arma::mat tau_eta = arma::ones(3,2);

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

  for(int i = 0; i < 250; i++){
    BayesFMMM::updateEta(y_obs, B_obs, tau_eta, Phi, xi, nu, Z, chi, sigma_sq,
                         i, 250, P, X, b_1, B_1, eta_samp);
  }

  arma::vec eta_ph = arma::zeros(150);
  arma::cube eta_est = arma::zeros(8,2,3);
  for(int j = 0; j < 8; j++){
    for(int l = 0; l < 2; l++){
      for(int d = 0; d < 3; d++){
        for(int k = 100; k < 250; k++){
          eta_ph(k - 100) = eta_samp(k,0)(j,l,d);
        }
        eta_est(j,l,d) = arma::median(eta_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = eta;
  mod(1,0) = eta_est;

  return mod;
}

// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdateEtaTempered(){
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

  // Make eta variables
  arma::cube eta(8, 2, 3);
  for(int i = 0; i < 3; i++){
    eta.slice(i) = arma::randn<arma::mat>(8,2);
  }

  // Make xi variables
  arma::field<arma::cube> xi(250,3);
  for(int i = 0; i < 3; i++){
    xi(0,i) = arma::zeros(8,2,2);
    for(int j =0; j < 2; j++){
      xi(0,i).slice(j) = (2-i) * 0.1 * arma::randn<arma::mat>(8,2);
    }
  }
  for(int i = 1; i < 250; i++){
    for(int j = 0; j < 3; j++){
      xi(i,j) = xi(0,j);
    }
  }

  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(40, 2, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(40, 3);
  arma::mat alpha(40,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::mat X = arma::randn<arma::mat>(40,2);
  arma::field<arma::vec> y_obs(40, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * (nu.row(l).t() + (eta.slice(l) * X.row(j).t()));
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * (Phi.slice(m).row(l).t() + (xi(0,l).slice(m)* X.row(j).t()));
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  // Initialize pi
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> eta_samp(250,1);

  for(int i = 0; i < 250; i++){
    eta_samp(i,0) = arma::zeros(8,2,3);
  }

  arma::mat tau_eta = arma::ones(3,2);

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

  for(int i = 0; i < 250; i++){
    BayesFMMM::updateEtaTempered(0.5, y_obs, B_obs, tau_eta, Phi, xi, nu, Z, chi,
                                 sigma_sq, i, 250, P, X, b_1, B_1, eta_samp);
  }

  arma::vec eta_ph = arma::zeros(150);
  arma::cube eta_est = arma::zeros(8,2,3);
  for(int j = 0; j < 8; j++){
    for(int l = 0; l < 2; l++){
      for(int d = 0; d < 3; d++){
        for(int k = 100; k < 250; k++){
          eta_ph(k - 100) = eta_samp(k,0)(j,l,d);
        }
        eta_est(j,l,d) = arma::median(eta_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = eta;
  mod(1,0) = eta_est;

  return mod;
}


// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdateEtaMV(){

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

  // Make eta variables
  arma::cube eta(8, 2, 3);
  for(int i = 0; i < 3; i++){
    eta.slice(i) = arma::randn<arma::mat>(8,2);
  }

  // Make xi variables
  arma::field<arma::cube> xi(250,3);
  for(int i = 0; i < 3; i++){
    xi(0,i) = arma::zeros(8,2,2);
    for(int j =0; j < 2; j++){
      xi(0,i).slice(j) = (2-i) * 0.1 * arma::randn<arma::mat>(8,2);
    }
  }
  for(int i = 1; i < 250; i++){
    for(int j = 0; j < 3; j++){
      xi(i,j) = xi(0,j);
    }
  }

  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(40, 2, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(40, 3);
  arma::mat alpha(40,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::mat X = arma::randn<arma::mat>(40,2);
  arma::mat y_obs = arma::zeros(40, 8);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * (nu.row(l).t() + (eta.slice(l) * X.row(j).t()));
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * (Phi.slice(m).row(l).t() + (xi(0,l).slice(m)* X.row(j).t()));
      }
    }
    y_obs.row(j) = arma::mvnrnd(mean, sigma_sq *
      arma::eye(mean.n_elem, mean.n_elem)).t();
  }

  // Initialize pi
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> eta_samp(250,1);

  for(int i = 0; i < 250; i++){
    eta_samp(i,0) = arma::zeros(8,2,3);
  }

  arma::mat tau_eta = arma::ones(3,2);

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

  for(int i = 0; i < 250; i++){
    BayesFMMM::updateEtaMV(y_obs, tau_eta, Phi, xi, nu, Z, chi, sigma_sq,
                           i, 250, P, X, b_1, B_1, eta_samp);
  }

  arma::vec eta_ph = arma::zeros(150);
  arma::cube eta_est = arma::zeros(8,2,3);
  for(int j = 0; j < 8; j++){
    for(int l = 0; l < 2; l++){
      for(int d = 0; d < 3; d++){
        for(int k = 100; k < 250; k++){
          eta_ph(k - 100) = eta_samp(k,0)(j,l,d);
        }
        eta_est(j,l,d) = arma::median(eta_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = eta;
  mod(1,0) = eta_est;

  return mod;
}


context("Unit tests for Eta parameters") {
  test_that("Sampler for Eta parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateEta();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int l = 0; l < est.n_slices; l++){
          if(std::abs(est(i,j,l) - truth(i,j,l)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Eta parameters under tempered likelihood"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateEtaTempered();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int l = 0; l < est.n_slices; l++){
          if(std::abs(est(i,j,l) - truth(i,j,l)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Eta parameters for the multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateEtaMV();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int l = 0; l < est.n_slices; l++){
          if(std::abs(est(i,j,l) - truth(i,j,l)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

}

