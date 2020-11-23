#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "computeMM.h"
#include "Distributions.H"

//' Tests updating Z
//'
//' @export
// [[Rcpp::export]]
arma::field<arma::vec> TestUpdateZ(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 0.01, .99);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat { bspline.basis(true)};
  // Make S_obs
  arma::field<arma::mat> S_obs(100,1);

  // set_space of unobserved time points
  arma::vec t_star = arma::regspace(0.005, 0.05, 1.005);
  splines2::BSpline bspline1;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline1 = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat1 { bspline1.basis(true)};
  // Make S_star
  arma::field<arma::mat> S_star(100,1);


  for(int i = 0; i < 100; i++)
  {
    S_obs(i,0) = bspline_mat;
    S_star(i,0) = bspline_mat1;
  }

  // Make nu matrix
  arma::mat Nu(3,8);
  Nu = {{2, 0, 1, 0, 0, 0, 1, 3},
        {1, 3, 0, 2, 0, 0, 3, 0},
        {5, 2, 5, 0, 3, 4, 1, 0}};
  Nu = Nu.t();

  // Set random seed
  arma::arma_rng::set_seed(123);

  // Make Phi matrix
  arma::mat x = arma::randu<arma::mat>(8,8);
  arma::cube Phi(8,8,3);
  Phi.slice(0) = x.t() * x;
  x = arma::randu<arma::mat>(8,8);
  Phi.slice(1) = x.t() * x;
  x = arma::randu<arma::mat>(8,8);
  Phi.slice(2) = x.t() * x;

  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  Z.col(0) = arma::vec(100, arma::fill::ones);

  // Initialize M,m, tilde_M, tilde_m
  arma::field<arma::mat> M(100,1);
  arma::field<arma::vec> m(100,1);
  arma::field<arma::mat> tilde_M(100,1);
  arma::field<arma::vec> tilde_m(100,1);
  for(int i = 0; i < 100; i ++)
  {
    M(i,0) = arma::zeros(100, 100);
    m(i,0) = arma::zeros(100);
    tilde_M(i,0) = arma::zeros(21, 21);
    tilde_m(i,0) = arma::zeros(21);
  }
  // Compute M matrices
  computeM(S_obs, Z, Phi, M);
  compute_m(S_obs, Z, Phi, Nu, m);

   arma::field<arma::vec> f_obs(100, 1);
  for(int j = 0; j < 100; j++)
  {
    f_obs(j,0) = Rmvnormal(M(j,0) * m(j,0), M(j,0));
  }
  return(f_obs);
}



