#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "computeMM.h"
#include "Distributions.H"
#include "UpdateClassMembership.H"
#include "UpdatePi.H"
#include "CalculateCov.H"

//' Tests updating Z
//'
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateZ(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 10, 990);
  arma::vec t_star = arma::regspace(0, 50, 950);
  arma::vec t_comb = arma::zeros(t_obs.n_elem + t_star.n_elem);
  t_comb.subvec(0, t_obs.n_elem - 1) = t_obs;
  t_comb.subvec(t_obs.n_elem, t_obs.n_elem + t_star.n_elem - 1) = t_star;
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_comb, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat { bspline.basis(true)};
  // Make S_obs
  arma::field<arma::mat> S_obs(100,1);

  arma::field<arma::mat> S_star(100,1);


  for(int i = 0; i < 100; i++)
  {
    S_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
    S_star(i,0) =  bspline_mat.submat(t_obs.n_elem, 0, t_obs.n_elem + t_star.n_elem - 1, 7);
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
        {1, 3, 0, 2, 0, 0, 3, 0},
        {5, 2, 5, 0, 3, 4, 1, 0}};
  nu = nu.t();

  // Make Phi matrix
  arma::cube Phi(8,8,3);
  for(int i=0; i < 3; i++)
  {
    Phi.slice(i) = 0.1 * arma::randu<arma::mat>(8,8);
  }
  // Rho
  // P = 8 => 8*9/2 = 36
  // K = 3 => 3*2/2 = 3
  arma::mat Rho(36, 3, arma::fill::randu);

  //initialize covariance placeholder
  arma::mat Cov(8,8, arma::fill::zeros);
  int rank = 8;

  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  Z.col(0) = arma::vec(100, arma::fill::ones);

  // Initialize mp_inv, mean_ph_obs, mean_ph_star
  arma::field<arma::mat> mp_inv(100,1);
  arma::field<arma::vec> mean_ph_obs(100,1);
  arma::field<arma::vec> mean_ph_star(100,1);
  arma::field<arma::vec> m_ph(100,1);
  arma::field<arma::mat> M_UV_ph(100,2);
  arma::field<arma::vec> M_S_ph(100,2);
  arma::field<arma::mat> M_UV(100,2);
  arma::field<arma::vec> M_S(100,2);
  arma::field<arma::mat> mean_UV(100,2);
  arma::field<arma::vec> mean_S(100,2);
  arma::field<arma::mat> UV_big(100,2);
  arma::field<arma::mat> UV_small(100,2);
  arma::field<arma::vec> S_big(100,2);

  for(int i = 0; i < 100; i++)
  {
    // dim of number of unobserved + observed time points
    mp_inv(i,0) = arma::zeros(120, 120);

    // dim of number of observed time points
    mean_ph_obs(i,0) = arma::zeros(100);

    // dim of number of unobserved time points
    mean_ph_star(i,0) = arma::zeros(20);

    // dim of number of unobserved time points
    m_ph(i,0) = arma::zeros(20);

    // dim of number of unobserved time points x rank
    M_UV_ph(i,0) = arma::zeros(20, 8);
    M_UV_ph(i,1) = arma::zeros(20, 8);
    M_UV(i,0) = arma::zeros(20, 8);
    M_UV(i,1) = arma::zeros(20, 8);

    // dim number of observed time points x observed time points
    mean_UV(i,0) = arma::zeros(100, 100);
    mean_UV(i,1) = arma::zeros(100, 100);
    mean_S(i,0) = arma::zeros(100);
    mean_S(i,1) = arma::zeros(100);

    // dim of rank
    M_S_ph(i,0) = arma::zeros(8);
    M_S_ph(i,1) = arma::zeros(8);
    M_S(i,0) = arma::zeros(8);
    M_S(i,1) = arma::zeros(8);

    // dim number of unobserved time points + observed time points
    UV_big(i,0) = arma::zeros(120, 120);
    UV_big(i,1) = arma::zeros(120, 120);
    S_big(i,0) = arma::zeros(120);
    S_big(i,1) = arma::zeros(120);

    UV_small(i,0)= arma::zeros(20, 20);
    UV_small(i,1) = arma::zeros(20, 20);
  }

  arma::field<arma::vec> f_obs(100, 1);
  arma::vec mean = arma::zeros(100);
  arma::vec mean1 = arma::zeros(100);
  for(int j = 0; j < 100; j++)
  {
    mean = arma::zeros(100);
    for(int l = 0; l < 3; l++)
    {
      mean = mean + Z(j,l) * S_obs(j,0) * nu.col(l);
    }
    if(j == 1)
    {
      mean1 = mean;
    }
    getCov(Z.row(j), Phi, Rho, Cov);
    f_obs(j,0) = Rmvnormal(mean, S_obs(j,0) * Cov * S_obs(j,0).t(), rank);
  }

  // Initialize M, m
  // arma::field<arma::mat> M(100,1);
  arma::field<arma::vec> m(100,1);
  for(int i = 0; i < 100; i ++)
  {
    //M(i,0) = arma::zeros(100, 100);
    m(i,0) = arma::zeros(100);
  }
  // Compute M and m matrices
  compute_M_m(S_obs, S_star, f_obs, Z, Phi, Rho, nu, rank, Cov, mp_inv, mean_ph_obs,
             mean_ph_star, m, UV_big, UV_small, S_big, M_UV, M_S);

  // Compute f_star
  arma::field<arma::vec> f_star(100, 1);
  for(int i = 0; i < 100; i++)
  {
    f_star(i,0) = Rmvnormal(M_UV(i,1) * arma::diagmat(M_S(i,0)) * M_UV(i,0).t() * m(i,0), M_UV(i,1) * arma::diagmat(M_S(i,0)) * M_UV(i,0).t(), rank);
  }

  // Initialize pi
  arma::vec pi = {0.95, 0.5, 0.5};

  // Initialize placeholder
  arma::mat Z_ph = arma::zeros(100, 3);

  //Initialize Z_samp
   arma::cube Z_samp = arma::ones(100, 3, 100);
  for(int i = 0; i < 100; i++)
  {
    // updateZ(f_obs, f_star, pi, i, S_obs, S_star, Phi, Rho, nu, Cov, M, M_ph, m, m_ph,
    //        mean_ph_obs, mean_ph_star, Z_ph, mp_inv, Z_samp);
  }

  // arma::vec lpdf_true = arma::zeros(100);
  // arma::vec lpdf_false = arma::zeros(100);
  // arma::mat Z_ph1 = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  // Z_ph1.col(0) = arma::vec(100, arma::fill::ones);
  // arma::vec ss_true = arma::zeros(100);
  // arma::vec ss_false = arma::zeros(100);
  // double tr_M = 0;
  // for(int i = 0; i < 100; i++)
  // {
  //   compute_mi_Mi(S_obs, S_star, f_obs, Z_ph1, Phi, Rho, nu, i, Cov, mp_inv(i,0),
  //                 mean_ph_obs(i,0), mean_ph_star(i,0), m_ph(i,0), M_ph(i,0));
  //   getCov(Z.row(i), Phi, Rho, Cov);
  //   lpdf_true(i) = lpdf_z(M(i,0), m(i,0), f_obs(i,0), f_star(i,0), S_obs(i,0),
  //                    Cov, nu, pi, Z, i, mp_inv(i,0), mean_ph_obs(i,0));
  //
  //   //try svd
  //   mean_ph_obs(i,0).zeros();
  //   for(int l = 0; l < nu.n_cols; l++){
  //     mean_ph_obs(i,0) = mean_ph_obs(i,0) + Z(i,l) * S_obs(i,0) * nu.col(l);
  //   }
  //   mp_inv(i,0).submat(0, 0, S_obs.n_rows - 1, S_obs.n_rows - 1) = arma::symmatu(arma::pinv(S_obs(i,0) * Cov * S_obs(i, 0).t(), 1e-20 * arma::datum::eps));
  //   for(int j = 0; j < S_obs.n_rows; j++)
  //   {
  //     if(mp_inv(i,0)(j,j) < 0)
  //     {
  //       mp_inv(i, 0)(j,j) = 0;
  //     }
  //   }
  //   ss_true(i) = arma::dot(mp_inv(i,0).submat(0, 0, S_obs.n_rows - 1, S_obs.n_rows - 1) * (f_obs(i,0) - mean_ph_obs(i,0)),
  //             f_obs(i,0) - mean_ph_obs(i,0));
  //   getCov(Z_ph1.row(i), Phi, Rho, Cov);
  //   lpdf_false(i) = lpdf_z(M_ph(i,0), m_ph(i,0), f_obs(i,0), f_star(i,0),
  //                       S_obs(i,0), Cov, nu, pi, Z_ph1, i, mp_inv(i,0), mean_ph_obs(i,0));
  //
  //   mean_ph_obs(i,0).zeros();
  //   for(int l = 0; l < nu.n_cols; l++){
  //     mean_ph_obs(i,0) = mean_ph_obs(i,0) + Z_ph1(i,l) * S_obs(i,0) * nu.col(l);
  //   }
  //   mp_inv(i,0).submat(0, 0, S_obs.n_rows - 1, S_obs.n_rows - 1) = arma::symmatu(arma::pinv(S_obs(i,0) * Cov * S_obs(i, 0).t(), 1e-20 * arma::datum::eps));
  //   for(int j = 0; j < S_obs.n_rows; j++)
  //   {
  //     if(mp_inv(i,0)(j,j) < 0)
  //     {
  //       mp_inv(i, 0)(j,j) = 0;
  //     }
  //   }
  //   ss_false(i) = arma::dot(mp_inv(i,0).submat(0, 0, S_obs.n_rows - 1, S_obs.n_rows - 1) * (f_obs(i,0) - mean_ph_obs(i,0)),
  //                       f_obs(i,0) - mean_ph_obs(i,0));
  // }

  int n1 = 100;
  int n2 = 20;
  getCov(Z.row(0), Phi, Rho, Cov);
  mp_inv(0, 0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(0, 0) * Cov *
    S_obs(0, 0).t();
  mp_inv(0, 0).submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(0, 0) * Cov *
    S_star(0, 0).t();
  mp_inv(0, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(0, 0).submat(0, n1, n1 - 1,
         n1 + n2 - 1).t();
  mp_inv(0, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(0, 0) * Cov *
    S_star(0, 0).t();

  // arma::mat M_ph = arma::zeros(20,20);
  // arma::pinv(mp_inv(0, 0), mp_inv(0, 0));
  // arma::pinv(M_ph, mp_inv(0, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1));
  //
  // int i = 0;
  // mean_ph_obs(i, 0).zeros();
  // mean_ph_star(i, 0).zeros();
  // for(int j = 0; j < nu.n_cols; j ++){
  //   mean_ph_obs(i, 0) = mean_ph_obs(i, 0) + Z(i,j) * S_obs(i, 0) * nu.col(j);
  //   mean_ph_star(i, 0) = mean_ph_star(i, 0) + Z(i,j) * S_star(i, 0) *
  //     nu.col(j);
  // }
  //
  // arma::vec m_ph1 = arma::zeros(20);
  // // Compute mean
  // m_ph1 = mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs(i, 0)
  //                                                               - f_obs(i, 0)) +
  //     mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star(i, 0);

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Cov", Cov),
                                      Rcpp::Named("Z",Z),
                                      Rcpp::Named("f_obs", f_obs),
                                      Rcpp::Named("f_star", f_star),
                                      Rcpp::Named("M", M_UV(0,1) * arma::diagmat(M_S(0,0)) * M_UV(0,0).t()),
                                      Rcpp::Named("M-arma",mp_inv(0,0)),
                                      Rcpp::Named("m", m),
                                      Rcpp::Named("S_obs",S_obs),
                                      Rcpp::Named("S_star",S_star),
                                      Rcpp::Named("mp_inv_V", M_UV(0,1)),
                                      Rcpp::Named("mp_inv_S", M_S(0,0)),
                                      Rcpp::Named("mp_inv_S_inv", M_S(0,1)));
  return mod;
}

// //' Tests updating Z with no unobserved data points
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateZNoUnobs(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat { bspline.basis(true)};
//   // Make S_obs
//   arma::field<arma::mat> S_obs(100,1);
//
//
//   for(int i = 0; i < 100; i++)
//   {
//     S_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//   nu = nu.t();
//
//   // Make Phi matrix
//   arma::cube Phi(8,7,3);
//   for(int i=0; i < 3; i++)
//   {
//     Phi.slice(i) = 0.1 * arma::randu<arma::mat>(8,7);
//   }
//   // Rho
//   // P = 8 => 8*9/2 = 36
//   // K = 3 => 3*2/2 = 3
//   arma::mat Rho(36, 3, arma::fill::randu);
//
//   //initialize covariance placeholder
//   arma::mat Cov(8,8, arma::fill::zeros);
//
//   // Make Z matrix
//   arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   Z.col(0) = arma::vec(100, arma::fill::ones);
//
//   // mean_ph_obs
//   arma::field<arma::vec> mean_ph_obs(100,1);
//   arma::field<arma::mat> ph(100, 1);
//   for(int i = 0; i < 100; i++)
//   {
//
//     // dim of number of observed time points
//     mean_ph_obs(i,0) = arma::zeros(100);
//     ph(i,0) = arma::zeros(100, 100);
//   }
//
//   arma::field<arma::vec> f_obs(100, 1);
//   arma::vec mean = arma::zeros(100);
//   for(int j = 0; j < 100; j++)
//   {
//     mean = arma::zeros(100);
//     for(int l = 0; l < 3; l++)
//     {
//       mean = mean + Z(j,l) * S_obs(j,0) * nu.col(l);
//     }
//     getCov(Z.row(j), Phi, Rho, Cov);
//     f_obs(j,0) = Rmvnormal(mean, S_obs(j,0) * Cov * Cov.t() * S_obs(j,0).t());
//   }
//
//   // Initialize pi
//   arma::vec pi = {0.95, 0.5, 0.5};
//
//   // Initialize placeholder
//   arma::mat Z_ph = arma::zeros(100, 3);
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(100, 3, 100);
//
//
//   for(int i = 0; i < 100; i++)
//   {
//     updateZ(f_obs, pi, i, S_obs, Phi, Rho, nu, ph, Cov,
//             mean_ph_obs, Z_ph, Z_samp);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
//                                       Rcpp::Named("Z",Z));
//   return mod;
// }
//
// //' Tests updating Pi
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdatePi()
// {
//   // Make Z matrix
//   arma::mat Z = arma::randi<arma::mat>(1000, 3, arma::distr_param(0,1));
//   double alpha = 1;
//   arma::mat pi = arma::zeros(100, 3);
//
//   for(int i = 0; i < 100; i ++)
//   {
//     update_pi(alpha, i, Z,  pi);
//   }
//   arma::vec prob = arma::zeros(3);
//   for(int i = 0; i < 3; i++)
//   {
//     prob(i) = arma::accu(Z.col(i)) / 1000;
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("pi", pi),
//                                       Rcpp::Named("prob", prob));
//   return mod;
// }
//
// //' Tests updating Z using single covariance matrix
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateZSingleMat(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 1, 99);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat { bspline.basis(true)};
//   // Make S_obs
//   arma::field<arma::mat> S_obs(100,1);
//
//   // set_space of unobserved time points
//   arma::vec t_star = arma::regspace(0.5, 5, 95.5);
//   splines2::BSpline bspline1;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline1 = splines2::BSpline(t_star, 8);
//   // Get Basis matrix (20 x 8)
//   arma::mat bspline_mat1 {bspline1.basis(true)};
//   // Make S_star
//   arma::field<arma::mat> S_star(100,1);
//
//
//   for(int i = 0; i < 100; i++)
//   {
//     S_obs(i,0) = bspline_mat;
//     S_star(i,0) = bspline_mat1;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//   nu = nu.t();
//
//   // Set random seed
//   arma::arma_rng::set_seed(123);
//
//   // Make Phi matrix
//   arma::mat Phi(8,8);
//   arma::mat x = arma::randu<arma::mat>(8,8);
//   Phi = x.t() * x;
//
//   // Initialize Cov matrix
//   arma::mat Cov(8,8, arma::fill::zeros);
//
//   // Make Z matrix
//   arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   Z.col(0) = arma::vec(100, arma::fill::ones);
//
//   // Initialize mp_inv, mean_ph_obs, mean_ph_star
//   arma::field<arma::mat> mp_inv(100,1);
//   arma::field<arma::vec> mean_ph_obs(100,1);
//   arma::field<arma::vec> mean_ph_star(100,1);
//   arma::field<arma::vec> m_ph(100,1);
//   arma::field<arma::mat> M_ph(100,1);
//
//   for(int i = 0; i < 100; i++)
//   {
//     // dim of number of unobserved + observed time points
//     mp_inv(i,0) = arma::zeros(120, 120);
//
//     // dim of number of observed time points
//     mean_ph_obs(i,0) = arma::zeros(100);
//
//     // dim of number of unobserved time points
//     mean_ph_star(i,0) = arma::zeros(20);
//
//     // dim of number of unobserved time points
//     m_ph(i,0) = arma::zeros(20);
//
//     // dim of number of unobserved time points
//     M_ph(i,0) = arma::zeros(20, 20);
//   }
//
//   arma::field<arma::vec> f_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//   for(int j = 0; j < 100; j++)
//   {
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++)
//     {
//       mean = mean + Z(j,l) * S_obs(j,0) * nu.col(l);
//     }
//     f_obs(j,0) = Rmvnormal(mean, Phi);
//   }
//
//   // Initialize M, m
//   arma::field<arma::mat> M(100,1);
//   arma::field<arma::vec> m(100,1);
//   for(int i = 0; i < 100; i ++)
//   {
//     M(i,0) = arma::zeros(100, 100);
//     m(i,0) = arma::zeros(100);
//   }
//   // Compute M and m matrices
//   compute_M_m(S_obs, S_star, f_obs, Z, Phi, nu, mp_inv, mean_ph_obs,
//               mean_ph_star, m, M);
//
//   // Compute f_star
//   arma::field<arma::vec> f_star;
//   for(int i = 0; i < 100; i++)
//   {
//     f_star(i,0) = Rmvnormal(M(i,0) * m(i,0), M(i,0));
//   }
//
//   // Initialize pi
//   arma::vec pi = {0.95, 0.5, 0.5};
//
//   // Initialize placeholder
//   arma::mat Z_ph = arma::zeros(100, 3);
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(100, 3, 100);
//   for(int i = 0; i < 100; i++)
//   {
//     updateZ(f_obs, f_star, pi, i, S_obs, S_star, Phi, nu, Cov, M, M_ph, m, m_ph,
//             mean_ph_obs, mean_ph_star, Z_ph, mp_inv, Z_samp);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
//                                       Rcpp::Named("Z", Z));
//   return mod;
// }
//
//
// //' Tests updating Z using single covariance matrix with no unobserved Observations
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateZSingleMatNoUnobs(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 1, 99);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat { bspline.basis(true)};
//   // Make S_obs
//   arma::field<arma::mat> S_obs(100,1);
//
//   for(int i = 0; i < 100; i++)
//   {
//     S_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//   nu = nu.t();
//
//   // Set random seed
//   arma::arma_rng::set_seed(123);
//
//   // Make Phi matrix
//   arma::mat Phi(8,8);
//   arma::mat x = arma::randu<arma::mat>(8,8);
//   Phi = x.t() * x;
//
//   // Initialize Cov matrix
//   arma::mat Cov(8,8, arma::fill::zeros);
//
//   // Make Z matrix
//   arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   Z.col(0) = arma::vec(100, arma::fill::ones);
//
//   // mean_ph_obs
//   arma::field<arma::vec> mean_ph_obs(100,1);
//   arma::field<arma::mat> ph(100,1);
//   for(int i = 0; i < 100; i++)
//   {
//     // dim of number of observed time points
//     mean_ph_obs(i,0) = arma::zeros(100);
//     ph(i,0) = arma::zeros(100, 100);
//   }
//
//   arma::field<arma::vec> f_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//   for(int j = 0; j < 100; j++)
//   {
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++)
//     {
//       mean = mean + Z(j,l) * S_obs(j,0) * nu.col(l);
//     }
//     f_obs(j,0) = Rmvnormal(mean, Phi);
//   }
//
//   // Initialize pi
//   arma::vec pi = {0.95, 0.5, 0.5};
//
//   // Initialize placeholder
//   arma::mat Z_ph = arma::zeros(100, 3);
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(100, 3, 100);
//   for(int i = 0; i < 100; i++)
//   {
//     updateZ(f_obs, pi, i, S_obs, Phi, nu, ph, Cov,
//             mean_ph_obs, Z_ph, Z_samp);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
//                                       Rcpp::Named("Z", Z));
//   return mod;
// }
//
//
//


