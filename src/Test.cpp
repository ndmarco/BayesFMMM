#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "computeMM.h"
#include "Distributions.H"
#include "UpdateClassMembership.H"
#include "UpdatePi.H"

//' Tests updating Z
//'
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateZ(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 1, 99);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat { bspline.basis(true)};
  // Make S_obs
  arma::field<arma::mat> S_obs(100,1);

  // set_space of unobserved time points
  arma::vec t_star = arma::regspace(0.5, 5, 95.5);
  splines2::BSpline bspline1;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline1 = splines2::BSpline(t_star, 8);
  // Get Basis matrix (20 x 8)
  arma::mat bspline_mat1 {bspline1.basis(true)};
  // Make S_star
  arma::field<arma::mat> S_star(100,1);


  for(int i = 0; i < 100; i++)
  {
    S_obs(i,0) = bspline_mat;
    S_star(i,0) = bspline_mat1;
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
        {1, 3, 0, 2, 0, 0, 3, 0},
        {5, 2, 5, 0, 3, 4, 1, 0}};
  nu = nu.t();

  // Set random seed
  arma::arma_rng::set_seed(123);

  // Make phi matrix
  arma::cube phi(8,7,7);
  for(int i=0; i < 7; i++)
  {
    phi.slice(i) = arma::randu<arma::mat>(8,7);
  }
  // Create mapping
  std::map<double, int> Map;
  for(int i=0; i<7; i++)
  {
    Map[i+1.0] = i;
  }

  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  Z.col(0) = arma::vec(100, arma::fill::ones);

  // Initialize mp_inv, mean_ph_obs, mean_ph_star
  arma::field<arma::mat> mp_inv(100,1);
  arma::field<arma::vec> mean_ph_obs(100,1);
  arma::field<arma::vec> mean_ph_star(100,1);
  arma::field<arma::vec> m_ph(100,1);
  arma::field<arma::mat> M_ph(100,1);

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

    // dim of number of unobserved time points
    M_ph(i,0) = arma::zeros(20, 20);
  }

  arma::field<arma::vec> f_obs(100, 1);
  arma::vec mean = arma::zeros(100);
  for(int j = 0; j < 100; j++)
  {
    mean = arma::zeros(100);
    for(int l = 0; l < 3; l++)
    {
      mean = mean + Z(j,l) * S_obs(j,0) * nu.col(l);
    }
    f_obs(j,0) = Rmvnormal(mean, S_obs(j,0) * phi.slice(Map.at(get_ind(Z.row(j).t()))) *
      phi.slice(Map.at(get_ind(Z.row(j).t()))).t() * S_obs(j,0).t());
  }

  // Initialize M, m
  arma::field<arma::mat> M(100,1);
  arma::field<arma::vec> m(100,1);
  for(int i = 0; i < 100; i ++)
  {
    M(i,0) = arma::zeros(100, 100);
    m(i,0) = arma::zeros(100);
  }
  // Compute M and m matrices
  compute_M_m(S_obs, S_star, f_obs, Z, phi, Map, nu, mp_inv, mean_ph_obs,
             mean_ph_star, m, M);

  // Compute f_star
  arma::field<arma::vec> f_star(100, 1);
  for(int i = 0; i < 100; i++)
  {
    f_star(i,0) = Rmvnormal(M(i,0) * m(i,0), M(i,0));
  }

  // Initialize pi
  arma::vec pi = {0.95, 0.5, 0.5};

  // Initialize placeholder
  arma::mat Z_ph = arma::zeros(100, 3);

  //Initialize Z_samp
   arma::cube Z_samp = arma::ones(100, 3, 100);
  for(int i = 0; i < 100; i++)
  {
    updateZ(f_obs, f_star, pi, i, S_obs, S_star, phi, nu, Map, M, M_ph, m, m_ph,
            mean_ph_obs, mean_ph_star, Z_ph, mp_inv, Z_samp);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
                                      Rcpp::Named("Z",Z));
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

//' Tests updating Z
//'
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateZSingleMat(){
  // Set space of functions
  arma::vec t_obs =  arma::regspace(0, 1, 99);
  splines2::BSpline bspline;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline = splines2::BSpline(t_obs, 8);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat { bspline.basis(true)};
  // Make S_obs
  arma::field<arma::mat> S_obs(100,1);

  // set_space of unobserved time points
  arma::vec t_star = arma::regspace(0.5, 5, 95.5);
  splines2::BSpline bspline1;
  // Create Bspline object with 8 degrees of freedom
  // 8 - 3 - 1 internal nodes
  bspline1 = splines2::BSpline(t_star, 8);
  // Get Basis matrix (20 x 8)
  arma::mat bspline_mat1 {bspline1.basis(true)};
  // Make S_star
  arma::field<arma::mat> S_star(100,1);


  for(int i = 0; i < 100; i++)
  {
    S_obs(i,0) = bspline_mat;
    S_star(i,0) = bspline_mat1;
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};
  nu = nu.t();

  // Set random seed
  arma::arma_rng::set_seed(123);

  // Make phi matrix
  arma::mat phi(8,8);
  arma::mat x = arma::randu<arma::mat>(8,8);
  phi = x.t() * x;

  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  Z.col(0) = arma::vec(100, arma::fill::ones);

  // Initialize mp_inv, mean_ph_obs, mean_ph_star
  arma::field<arma::mat> mp_inv(100,1);
  arma::field<arma::vec> mean_ph_obs(100,1);
  arma::field<arma::vec> mean_ph_star(100,1);
  arma::field<arma::vec> m_ph(100,1);
  arma::field<arma::mat> M_ph(100,1);

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

    // dim of number of unobserved time points
    M_ph(i,0) = arma::zeros(20, 20);
  }

  arma::field<arma::vec> f_obs(100, 1);
  arma::vec mean = arma::zeros(8);
  for(int j = 0; j < 100; j++)
  {
    for(int l = 0; l < 3; l++)
    {
      mean = mean + Z(j,l) * nu(l);
    }
    f_obs(j,0) = Rmvnormal(mean, phi);
  }

  // Initialize M, m
  arma::field<arma::mat> M(100,1);
  arma::field<arma::vec> m(100,1);
  for(int i = 0; i < 100; i ++)
  {
    M(i,0) = arma::zeros(100, 100);
    m(i,0) = arma::zeros(100);
  }
  // Compute M and m matrices
  compute_M_m(S_obs, S_star, f_obs, Z, phi, nu, mp_inv, mean_ph_obs,
              mean_ph_star, m, M);

  // Compute f_star
  arma::field<arma::vec> f_star;
  for(int i = 0; i < 100; i++)
  {
    f_star(i,0) = Rmvnormal(M(i,0) * m(i,0), M(i,0));
  }

  // Initialize pi
  arma::vec pi = {0.95, 0.5, 0.5};

  // Initialize placeholder
  arma::mat Z_ph = arma::zeros(100, 3);


  //Initialize Z_samp
  arma::cube Z_samp = arma::ones(100, 3, 100);
  for(int i = 0; i < 100; i++)
  {
    updateZ(f_obs, f_star, pi, i, S_obs, S_star, phi, nu, M, M_ph, m, m_ph,
            mean_ph_obs, mean_ph_star, Z_ph, mp_inv, Z_samp);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
                                      Rcpp::Named("Z", Z));
  return mod;
}






