#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "Distributions.H"
#include "UpdateClassMembership.H"
#include "UpdatePi.H"
#include "UpdatePhi.H"
#include "UpdateDelta.H"
#include "UpdateA.H"
#include "UpdateGamma.H"
#include "UpdateNu.H"
#include "UpdateTau.H"
#include "UpdateSigma.H"
#include "UpdateChi.H"

//' Tests updating Z
//'
//' @name TestUpdateZ
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
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
    y_star(j, 0).row(0) = arma::mvnrnd(B_star(j, 0) * mean, sigma_sq *
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
            sigma_sq, 0.6, i, 100, Z_ph, Z_samp);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
                                      Rcpp::Named("Z",Z),
                                      Rcpp::Named("f_obs", y_obs),
                                      Rcpp::Named("f_star", y_star));
  return mod;
}


//' Tests updating Pi
//'
//' @name TestUpdatePi
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdatePi()
{
  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(1000, 3, arma::distr_param(0,1));
  double alpha = 1;
  arma::mat pi = arma::zeros(3, 100);

  for(int i = 0; i < 100; i ++)
  {
    update_pi(alpha, Z, i, 100,  pi);
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
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
    y_star(j, 0).row(0) = arma::mvnrnd(B_star(j, 0) * mean, sigma_sq *
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
    updatePhi(y_obs, y_star, B_obs, B_star, nu, gamma, tilde_tau, Z, chi,
             sigma_sq, i, 100, m_1, M_1, Phi_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Phi", Phi),
                                      Rcpp::Named("Phi_samp", Phi_samp));
  return mod;
}


//' Tests updating Delta
//'
//' @name TestUpdateDelta
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateDelta(){
  // Specify hyperparameters
  arma::vec a_12 = {2, 2};
  // Make Delta vector
  arma::vec Delta = arma::zeros(5);
  for(int i=0; i < 5; i++){
    Delta(i) = R::rgamma(4, 1);
  }
  // Make Gamma cube
  arma::cube Gamma(3,8,5);
  for(int i=0; i < 5; i++){
    for(int j = 0; j < 3; j++){
      for(int k = 0; k < 8; k++){
        Gamma(j,k,i) =  R::rgamma(1.5, 1/1.5);
      }
    }
  }

  // Make Phi matrix
  arma::cube Phi(3,8,5);
  arma::mat delta = arma::ones(5,10000);
  for(int m = 0; m < 10000; m++){
    double tau  = 1;
    for(int i=0; i < 5; i++){
      tau = tau * Delta(i);
      for(int j=0; j < 3; j++){
        for(int k=0; k < 8; k++){
          Phi(j,k,i) = R::rnorm(0, (1/ std::pow(Gamma(j,k,i)*tau, 0.5)));
        }
      }
    }
    updateDelta(Phi, Gamma, a_12, m, 10000, delta);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("delta", Delta),
                                      Rcpp::Named("gamma", Gamma),
                                      Rcpp::Named("phi", Phi),
                                      Rcpp::Named("delta_samp", delta));
  return mod;
}

//' Tests updating A
//'
//' @name TestUpdateA
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateA(){
  double a_1 = 2;
  double a_2 = 3;
  arma::vec delta = arma::zeros(5);
  double alpha1 = 2;
  double beta1 = 1;
  double alpha2 = 3;
  double beta2 = 1;
  arma::mat A = arma::ones(2, 1000);
  for(int i = 0; i < 1000; i++){
    for(int j = 0; j < 5; j++){
      if(j == 0){
        delta(j) = R::rgamma(a_1, 1);
      }else{
        delta(j) = R::rgamma(a_2, 1);
      }
    }
    updateA(alpha1, beta1, alpha2, beta2, delta, sqrt(1), sqrt(1), i, 1000, A);
  }

  double lpdf_true = lpdf_a2(alpha2, beta2, 2.0, delta);
  double lpdf_false = lpdf_a2(alpha2, beta2, 1.0, delta);
  double lpdf_true1 = lpdf_a1(alpha1, beta1, 3.0, delta(0));
  double lpdf_false1 = lpdf_a1(alpha1, beta1, 2.0, delta(0));
  double sum = 0;
  for(int i = 1; i < delta.n_elem; i++){
    sum = sum + (0.05 - 1) * log(delta(i));
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("a_1", a_1),
                                      Rcpp::Named("a_2", a_2),
                                      Rcpp::Named("A", A),
                                      Rcpp::Named("delta", delta.n_elem - 1),
                                      Rcpp::Named("lpdf_1", (delta.n_elem - 1)),
                                      Rcpp::Named("lpdf_2", (alpha2 - 1) * log(0.05)),
                                      Rcpp::Named("lpdf_3", -(0.05 * beta2)),
                                      Rcpp::Named("lpdf_4", sum),
                                      Rcpp::Named("lpdf_true", lpdf_true),
                                      Rcpp::Named("lpdf_false", lpdf_false),
                                      Rcpp::Named("lpdf_true1", lpdf_true1),
                                      Rcpp::Named("lpdf_false1", lpdf_false1));
  return mod;
}

//' Tests updating Gamma
//'
//' @name TestUpdateGamma
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateGamma(){
  // Specify hyperparameters
  double nu = 3;
  // Make Delta vector
  arma::vec Delta = arma::zeros(5);
  for(int i=0; i < 5; i++){
    Delta(i) = R::rgamma(4, 1);
  }
  // Make Gamma cube
  arma::cube Gamma(3,8,5);
  for(int i=0; i < 5; i++){
    for(int j = 0; j < 3; j++){
      for(int k = 0; k < 8; k++){
        Gamma(j,k,i) =  R::rgamma(nu, 1/nu);
      }
    }
  }

  // Make Phi matrix
  arma::cube Phi(3,8,5);
  arma::field<arma::cube> gamma(1000,1);
  for(int i = 0; i < 1000; i++){
    gamma(i,0) = arma::zeros(3,8,5);
  }
  for(int m = 0; m < 1000; m++){
    double tau  = 1;
    for(int i=0; i < 5; i++){
      tau = tau * Delta(i);
      for(int j=0; j < 3; j++){
        for(int k=0; k < 8; k++){
          Phi(j,k,i) = R::rnorm(0, (1/ std::pow(Gamma(j,k,i)*tau, 0.5)));
        }
      }
    }
    updateGamma(nu, Delta, Phi, m, 1000, gamma);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("gamma", Gamma),
                                      Rcpp::Named("gamma_iter", gamma),
                                      Rcpp::Named("phi", Phi),
                                      Rcpp::Named("delta", Delta));
  return mod;
}

//' Tests updating Nu
//'
//' @name TestUpdateNu
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateNu(){
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
  double sigma_sq = 0.01;

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
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
    y_star(j, 0).row(0) = arma::mvnrnd(B_star(j, 0) * mean, sigma_sq *
      arma::eye(B_star(j,0).n_rows, B_star(j,0).n_rows)).t();
    for(int i = 1; i < 100; i++){
      y_star(j, 0).row(i) = y_star(j, 0).row(0);
    }
  }

  // Initialize pi
  arma::vec pi = {0.95, 0.5, 0.5};

  arma::cube Nu_samp(nu.n_rows, nu.n_cols, 100);
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
  tau = tau * 10;
  for(int i = 0; i < 100; i++){
    updateNu(y_obs, y_star, B_obs, B_star, tau, Phi, Z, chi, sigma_sq, i, 100,
             P, b_1, B_1, Nu_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu_samp", Nu_samp),
                                      Rcpp::Named("nu", nu));
  return mod;
}

//' Tests updating Tau
//'
//' @name TestUpdateTau
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateTau(){
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
  arma::mat tau_samp(1000, 6, arma::fill::zeros);
  arma::vec zeros_nu(100, arma::fill::zeros);
  for(int i = 0; i < 1000; i++){
    for(int j = 0; j < 6; j++){
      nu.row(j) = arma::mvnrnd(zeros_nu, arma::pinv(P * tau(j))).t();
    }
    updateTau(1, 1, nu, i, 1000, P, tau_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("tau_samp", tau_samp),
                                      Rcpp::Named("tau", tau));
  return mod;
}


//' Tests updating Sigma
//'
//' @name TestUpdateSigma
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateSigma(){
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
  double sigma_sq = 0.5;

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
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
    y_star(j, 0).row(0) = arma::mvnrnd(B_star(j, 0) * mean, sigma_sq *
      arma::eye(B_star(j,0).n_rows, B_star(j,0).n_rows)).t();
    for(int i = 1; i < 100; i++){
      y_star(j, 0).row(i) = y_star(j, 0).row(0);
    }
  }
  double alpha_0 = 1;
  double beta_0 = 1;
  arma::vec sigma_samp(100, arma::fill::zeros);
  for(int i = 0; i < 100; i++){
    updateSigma(y_obs, y_star, B_obs, B_star, alpha_0, beta_0, nu, Phi, Z, chi,
                i, 100, sigma_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("sigma_samp", sigma_samp),
                                      Rcpp::Named("sigma", sigma_sq));
  return mod;
}

//' Tests updating chi
//'
//' @name TestUpdateChi
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateChi(){
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
    Phi.slice(i) = (5-i) * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.5;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  Z.col(0) = arma::vec(100, arma::fill::ones);

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);



  arma::cube chi_samp(100, 5, 1000, arma::fill::zeros);
  chi_samp.slice(0) = chi;
  for(int i = 0; i < 1000; i++){
    for(int j = 0; j < 100; j++){
      y_star(j,0) = arma::zeros(1000, B_star(j,0).n_rows);
      mean = arma::zeros(8);
      for(int l = 0; l < 3; l++){
        mean = mean + Z(j,l) * nu.row(l).t();
        for(int m = 0; m < Phi.n_slices; m++){
          mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
        }
      }
      y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
        arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
      y_star(j, 0).row(0) = arma::mvnrnd(B_star(j, 0) * mean, sigma_sq *
        arma::eye(B_star(j,0).n_rows, B_star(j,0).n_rows)).t();
      for(int k = 1; k < 1000; k++){
        y_star(j, 0).row(k) = y_star(j, 0).row(0);
      }
    }
    updateChi(y_obs, y_star, B_obs, B_star, Phi, nu, Z, sigma_sq, i, 1000,
              chi_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("chi_samp", chi_samp),
                                      Rcpp::Named("chi", chi),
                                      Rcpp::Named("Z", Z));
  return mod;
}
