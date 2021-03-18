#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "Distributions.H"
#include "UpdateClassMembership.H"
#include "UpdatePi.H"
#include "UpdatePhi.H"

//' Tests updating Z
//'
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateZ()
{
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
  // Make B_obs
  arma::field<arma::mat> B_obs(100,1);

  arma::field<arma::mat> B_star(100,1);


  for(int i = 0; i < 100; i++)
  {
    B_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
    B_star(i,0) =  bspline_mat.submat(t_obs.n_elem, 0,
           t_obs.n_elem + t_star.n_elem - 1, 7);
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
  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  Z.col(0) = arma::vec(100, arma::fill::ones);

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    y_star(j,0) = arma::zeros(100, B_star(j,0).n_rows);
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = Rmvnormal(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
    y_star(j, 0).row(0) = Rmvnormal(B_star(j, 0) * mean, sigma_sq *
      arma::eye(B_star(j,0).n_rows, B_star(j,0).n_rows)).t();
    for(int i = 1; i < 100; i++){
      y_star(j, 0).row(i) = y_star(j, 0).row(0);
    }
  }

  // Initialize pi
  arma::vec pi = {0.95, 0.5, 0.5};

  // Initialize placeholder
  arma::mat Z_ph = arma::zeros(100, 3);

  //Initialize Z_samp
   arma::cube Z_samp = arma::ones(100, 3, 100);
  for(int i = 0; i < 100; i++)
  {
    updateZ(y_obs, y_star, B_obs, B_star, Phi, nu, chi, pi,
            sigma_sq, 0.6, i, Z_ph, Z_samp);
  }
  double lpdf_1 = lpdf_z(y_obs(0,0), y_star(0,0).row(1), B_obs(0,0), B_star(0,0),
                         Phi, nu, chi.row(0), pi, Z.row(0), sigma_sq);
  double lpdf_2 = lpdf_z(y_obs(0,0), y_star(0,0).row(1), B_obs(0,0), B_star(0,0),
                         Phi, nu, chi.row(0), pi, Z.row(1), sigma_sq);
  double lpdf_3 = lpdf_z(y_obs(0,0), y_star(0,0).row(1), B_obs(0,0), B_star(0,0),
                         Phi, nu, chi.row(0), pi, Z.row(2), sigma_sq);
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("lpdf1", lpdf_1),
                                      Rcpp::Named("lpdf2", lpdf_2),
                                      Rcpp::Named("lpdf3", lpdf_3),
                                      Rcpp::Named("Z_samp", Z_samp),
                                      Rcpp::Named("Z",Z),
                                      Rcpp::Named("f_obs", y_obs),
                                      Rcpp::Named("f_star", y_star));
  return mod;
}

//' Tests updating Pi
//'
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdatePi()
{
  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(1000, 3, arma::distr_param(0,1));
  double alpha = 1;
  arma::mat pi = arma::zeros(100, 3);

  for(int i = 0; i < 100; i ++)
  {
    update_pi(alpha, i, Z,  pi);
  }
  arma::vec prob = arma::zeros(3);
  for(int i = 0; i < 3; i++)
  {
    prob(i) = arma::accu(Z.col(i)) / 1000;
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("pi", pi),
                                      Rcpp::Named("prob", prob));
  return mod;
}

//' Tests updating Phi
//'
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdatePhi()
{
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
  // Make B_obs
  arma::field<arma::mat> B_obs(100,1);

  arma::field<arma::mat> B_star(100,1);


  for(int i = 0; i < 100; i++)
  {
    B_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
    B_star(i,0) =  bspline_mat.submat(t_obs.n_elem, 0,
           t_obs.n_elem + t_star.n_elem - 1, 7);
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
  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  Z.col(0) = arma::vec(100, arma::fill::ones);

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    y_star(j,0) = arma::zeros(100, B_star(j,0).n_rows);
    mean = arma::zeros(8);
    for(int l = 0; l < 3; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = Rmvnormal(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
    y_star(j, 0).row(0) = Rmvnormal(B_star(j, 0) * mean, sigma_sq *
      arma::eye(B_star(j,0).n_rows, B_star(j,0).n_rows)).t();
    for(int i = 1; i < 100; i++){
      y_star(j, 0).row(i) = y_star(j, 0).row(0);
    }
  }

  // Initialize pi
  arma::vec pi = {0.95, 0.5, 0.5};

  arma::field<arma::cube> Phi_samp(100, 1);
  for(int i = 0; i < 100 ; i++){
    Phi_samp(i,0) = arma::zeros(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::vec tilde_tau = {2, 2.5, 3, 5, 10};
  for(int i = 0; i < 100; i++){
    UpdatePhi(y_obs, y_star, B_obs, B_star, nu, gamma, tilde_tau, Z, chi,
             sigma_sq, i, 100, m_1, M_1, Phi_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Phi", Phi),
                                      Rcpp::Named("Phi_samp", Phi_samp));
  return mod;
}
