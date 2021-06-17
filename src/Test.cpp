#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include "Distributions.H"
#include "UpdateClassMembership.H"
#include "UpdatePartialMembership.H"
#include "UpdatePi.H"
#include "UpdatePhi.H"
#include "UpdateDelta.H"
#include "UpdateA.H"
#include "UpdateGamma.H"
#include "UpdateNu.H"
#include "UpdateTau.H"
#include "UpdateSigma.H"
#include "UpdateChi.H"
#include "UpdateYStar.H"
#include "BFPMM.H"
#include "EstimateInitialState.H"
#include "UpdateAlpha3.H"

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
    updatePi(alpha, Z, i, 100,  pi);
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
  arma::mat A = arma::ones(1000, 2);
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
  arma::vec log_lik = arma::zeros(100);
  for(int i = 0; i < 100; i++){
    updateNu(y_obs, y_star, B_obs, B_star, tau, Phi, Z, chi, sigma_sq, i, 100,
             P, b_1, B_1, Nu_samp);
    log_lik(i) = calcLikelihood(y_obs, y_star, B_obs, B_star, Nu_samp.slice(i),
            Phi, Z, chi,i, sigma_sq);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu_samp", Nu_samp),
                                      Rcpp::Named("nu", nu),
                                      Rcpp::Named("log_lik", log_lik));
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
  for(int i = 0; i < 10000; i++){
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
      for(int k = 1; k < 10000; k++){
        y_star(j, 0).row(k) = y_star(j, 0).row(0);
      }
    }
    updateChi(y_obs, y_star, B_obs, B_star, Phi, nu, Z, sigma_sq, i, 10000,
              chi_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("chi_samp", chi_samp),
                                      Rcpp::Named("chi", chi),
                                      Rcpp::Named("Z", Z));
  return mod;
}

//' Tests updating chi
//'
//' @name TestUpdateChi
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateYStar(){
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

  arma::field<arma::mat> y_star_samp(100, 1);

  for(int i = 0; i < 100; i++){
    y_star_samp(i,0) = arma::zeros(1000, B_star(i,0).n_rows);
  }

  for(int i = 0; i < 1000; i++){
    updateYStar(B_star, nu, Phi, Z, chi, sigma_sq, i, 1000, y_star_samp);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("y_star_samp", y_star_samp),
                                      Rcpp::Named("y_star", y_star),
                                      Rcpp::Named("y_obs", y_obs));
  return mod;
}

//' Tests BFOC function
//'
//' @name TestBFOC
//' @export
// [[Rcpp::export]]
Rcpp::List TestBFOC(int tot_mcmc_iters){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    //t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
    Phi.slice(i) = (5-i) * 0.5 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.01;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  for(int i = 0; i < 100; i++){
    while(arma::accu(Z.row(i)) == 0){
      Z.row(i) = arma::randi<arma::rowvec>(3, arma::distr_param(0,1));
    }
  }

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    y_star(j,0) = arma::zeros(tot_mcmc_iters, B_star(j,0).n_rows);
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
    for(int k = 1; k < tot_mcmc_iters; k++){
      y_star(j, 0).row(k) = y_star(j, 0).row(0);
    }
  }
  arma::vec a_12 = {2, 2};
  Rcpp::List mod1 = BFOC(y_obs, t_obs1, n_funct, 3, 8, 5, tot_mcmc_iters, t_star1, 3, 0.7,
                         1, 2, 3, 1, 1, sqrt(1), sqrt(1), 1, 1, 1, 1);

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                      Rcpp::Named("y_star", mod1["y_star"]),
                                      Rcpp::Named("chi", mod1["chi"]),
                                      Rcpp::Named("pi", mod1["pi"]),
                                      Rcpp::Named("A", mod1["A"]),
                                      Rcpp::Named("delta", mod1["delta"]),
                                      Rcpp::Named("sigma", mod1["sigma"]),
                                      Rcpp::Named("tau", mod1["tau"]),
                                      Rcpp::Named("gamma", mod1["gamma"]),
                                      Rcpp::Named("Phi", mod1["Phi"]),
                                      Rcpp::Named("Z", mod1["Z"]),
                                      Rcpp::Named("loglik", mod1["loglik"]),
                                      Rcpp::Named("y_obs", y_obs),
                                      Rcpp::Named("Phi_true", Phi),
                                      Rcpp::Named("Z_true", Z),
                                      Rcpp::Named("nu_true", nu));
  return mod;
}

//' Tests BFOC function
//'
//' @name TestBFOC
//' @export
// [[Rcpp::export]]
Rcpp::List TestBFOC_SS(int tot_mcmc_iters, const std::string directory,
                       const int r_stored_iters){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++)
  {
    Phi.slice(i) = (3-i) * 0.1 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi(100, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  for(int i = 0; i < 100; i++){
    while(arma::accu(Z.row(i)) == 0){
      Z.row(i) = arma::randi<arma::rowvec>(3, arma::distr_param(0,1));
    }
  }

  arma::mat known_Z = Z.submat(0, 0, 99, 2);

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    y_star(j,0) = arma::zeros(tot_mcmc_iters, B_star(j,0).n_rows);
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
    for(int k = 1; k < tot_mcmc_iters; k++){
      y_star(j, 0).row(k) = y_star(j, 0).row(0);
    }
  }
  arma::vec a_12 = {2, 2};
  Rcpp::List mod1 = BFOC_SS(known_Z, y_obs, t_obs1, n_funct, 3, 8, 3, tot_mcmc_iters,
                            r_stored_iters, t_star1, 3, 0.7, 1, 2, 3, 1, 1,
                            sqrt(1), sqrt(1), 1, 1, 1, 1, directory);

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                      Rcpp::Named("y_star", mod1["y_star"]),
                                      Rcpp::Named("chi", mod1["chi"]),
                                      Rcpp::Named("pi", mod1["pi"]),
                                      Rcpp::Named("A", mod1["A"]),
                                      Rcpp::Named("delta", mod1["delta"]),
                                      Rcpp::Named("sigma", mod1["sigma"]),
                                      Rcpp::Named("tau", mod1["tau"]),
                                      Rcpp::Named("gamma", mod1["gamma"]),
                                      Rcpp::Named("Phi", mod1["Phi"]),
                                      Rcpp::Named("Z", mod1["Z"]),
                                      Rcpp::Named("loglik", mod1["loglik"]),
                                      Rcpp::Named("y_obs", y_obs),
                                      Rcpp::Named("Phi_true", Phi),
                                      Rcpp::Named("Z_true", Z),
                                      Rcpp::Named("nu_true", nu));
  return mod;
}

//' Tests BFOC function
//'
//' @name TestBFOC
//' @export
// [[Rcpp::export]]
Rcpp::List TestBFOC_SS_nu_Z(int tot_mcmc_iters, const std::string directory,
                       const int r_stored_iters){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++)
  {
    Phi.slice(i) = (3-i) * 0.1 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi(100, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  for(int i = 0; i < 100; i++){
    while(arma::accu(Z.row(i)) == 0){
      Z.row(i) = arma::randi<arma::rowvec>(3, arma::distr_param(0,1));
    }
  }

  arma::mat known_Z = Z.submat(0, 0, 99, 2);

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    y_star(j,0) = arma::zeros(tot_mcmc_iters, B_star(j,0).n_rows);
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
    for(int k = 1; k < tot_mcmc_iters; k++){
      y_star(j, 0).row(k) = y_star(j, 0).row(0);
    }
  }
  arma::vec a_12 = {2, 2};
  Rcpp::List mod1 = BFOC_SS(known_Z, y_obs, t_obs1, n_funct, 3, 8, 3, tot_mcmc_iters,
                            r_stored_iters, t_star1, 3, 0.7, 1, 2, 3, 1, 1,
                            sqrt(1), sqrt(1), 1, 1, 1, 1, directory);

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                      Rcpp::Named("y_star", mod1["y_star"]),
                                      Rcpp::Named("chi", mod1["chi"]),
                                      Rcpp::Named("pi", mod1["pi"]),
                                      Rcpp::Named("A", mod1["A"]),
                                      Rcpp::Named("delta", mod1["delta"]),
                                      Rcpp::Named("sigma", mod1["sigma"]),
                                      Rcpp::Named("tau", mod1["tau"]),
                                      Rcpp::Named("gamma", mod1["gamma"]),
                                      Rcpp::Named("Phi", mod1["Phi"]),
                                      Rcpp::Named("Z", mod1["Z"]),
                                      Rcpp::Named("loglik", mod1["loglik"]),
                                      Rcpp::Named("y_obs", y_obs),
                                      Rcpp::Named("Phi_true", Phi),
                                      Rcpp::Named("Z_true", Z),
                                      Rcpp::Named("nu_true", nu));
  return mod;
}


//' Tests Reading Matrix
//'
//' @name TestReadMat
//' @export
// [[Rcpp::export]]
arma::mat TestReadMat(std::string directory){
  arma::mat B;
  B.load(directory + "Pi0.txt");
  return B;
}

//' Tests Reading Cube
//'
//' @name TestReadCube
//' @export
// [[Rcpp::export]]
arma::cube TestReadCube(std::string directory){
  arma::cube B;
  B.load(directory);
  return B;
}

//' Tests Reading Field
//'
//' @name TestReadField
//' @export
// [[Rcpp::export]]
arma::field<arma::cube> TestReadField(std::string directory){
  arma::field<arma::cube> B;
  B.load(directory);
  return B;
}

//' Tests BFOC function
//'
//' @name GetStuff
//' @export
// [[Rcpp::export]]
Rcpp::List GetStuff(double sigma_sq){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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

  arma::mat nu;
  nu.load("c:\\Projects\\BayesFPMM\\data\\nu.txt");


  // Make Phi matrix
  arma::cube Phi;
  Phi.load("c:\\Projects\\BayesFPMM\\data\\Phi.txt");
  // double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi;
  chi.load("c:\\Projects\\BayesFPMM\\data\\chi.txt");


  // Make Z matrix
  arma::mat Z;
  Z.load("c:\\Projects\\BayesFPMM\\data\\Z.txt");

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < nu.n_rows; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) =arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }


  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("y", y_obs),
                                      Rcpp::Named("B", B_obs),
                                      Rcpp::Named("Phi_true", Phi),
                                      Rcpp::Named("Z_true", Z),
                                      Rcpp::Named("nu_true", nu));
  return mod;
}

//' Tests BFOC function
//'
//' @name GetStuff
//' @export
// [[Rcpp::export]]
Rcpp::List TestEstimateInitialZ(){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
  arma::mat bspline_mat {bspline.basis(true)};
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
  arma::mat nu(3,8, arma::fill::randn);
  nu = 2 * nu;

  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++)
  {
    Phi.slice(i) = (3-i) * 0.1 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi(100, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  for(int i = 0; i < 100; i++){
    if(i < 10){
      Z.row(i) = {1, 0, 0};
    }else if(i < 20){
      Z.row(i) = {0, 1, 0};
    }else if(i < 30){
      Z.row(i) = {0, 0, 1};
    }

    while(arma::accu(Z.row(i)) == 0){
      Z.row(i) = arma::randi<arma::rowvec>(3, arma::distr_param(0,1));
    }
  }


  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < nu.n_rows; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  arma::mat theta = BasisExpansion(y_obs, B_obs, 100, 3, 8);

  arma::mat Z_est = ZInitialState(B_obs, theta, 50, 3, 100, 0.001);


  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z", Z),
                                      Rcpp::Named("z_est", Z_est),
                                      Rcpp::Named("y_obs", y_obs),
                                      Rcpp::Named("Z_true", Z),
                                      Rcpp::Named("nu_true", nu));
  return mod;
}


//' Tests BFOC function
//'
//' @name GetStuff
//' @export
// [[Rcpp::export]]
Rcpp::List TestEstimateInitial(const int tot_mcmc_iters, const int r_stored_iters,
                               const std::string directory){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
  arma::mat nu;
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};
;

  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++)
  {
    Phi.slice(i) = (3-i) * 0.1 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi(100, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  for(int i = 0; i < 100; i++){
    if(i < 10){
      Z.row(i) = {1, 0, 0};
    }else if(i < 20){
      Z.row(i) = {0, 1, 0};
    }else if(i < 30){
      Z.row(i) = {0, 0, 1};
    }

    while(arma::accu(Z.row(i)) == 0){
      Z.row(i) = arma::randi<arma::rowvec>(3, arma::distr_param(0,1));
    }
  }

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < nu.n_rows; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  // estimate B-spline expansion
  arma::mat theta = BasisExpansion(y_obs, B_obs, 100, 3, 8);
  //estimate Z matrix
  arma::mat Z_est = ZInitialState(B_obs, theta, 50, 3, 100, 0.001);
  // estimate sigma
  double sigma_est = SigmaInitialState(y_obs, B_obs, theta, 100);
  // estimate nu
  arma::mat nu_est = NuInitialState(B_obs, Z_est, theta, 100);
  // get rest of estimates
  Rcpp::List output = PhiChiInitialState(Z_est, y_obs, t_obs1, 100, 3, 8, 3,
                                         1000, 200, t_star1, 3, 0.7, 1, 2, 3,
                                         1, 1, sqrt(1), sqrt(1), 1, 1, 1, 1,
                                         nu_est, sigma_est);

  // start MCMC sampling
  Rcpp::List mod1 = BFOC_U(y_obs, t_obs1, n_funct, 3, 8, 3, tot_mcmc_iters,
                           r_stored_iters, t_star1, 3, 0.7, 1, 2, 3, 1, 1,
                           sqrt(1), sqrt(1), 1, 1, 1, 1, directory, Z_est,
                           output["A_est"], output["pi_est"], output["tau_est"],
                           output["delta_est"], nu_est, output["Phi_est"],
                           output["gamma_est"], output["chi_est"],
                           output["y_star_est"], sigma_est);

  Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
                                        Rcpp::Named("y_obs", y_obs),
                                        Rcpp::Named("nu_true", nu),
                                        Rcpp::Named("Phi_true", Phi),
                                        Rcpp::Named("nu", mod1["nu"]),
                                        Rcpp::Named("y_star", mod1["y_star"]),
                                        Rcpp::Named("chi", mod1["chi"]),
                                        Rcpp::Named("pi", mod1["pi"]),
                                        Rcpp::Named("A", mod1["A"]),
                                        Rcpp::Named("delta", mod1["delta"]),
                                        Rcpp::Named("sigma", mod1["sigma"]),
                                        Rcpp::Named("tau", mod1["tau"]),
                                        Rcpp::Named("gamma", mod1["gamma"]),
                                        Rcpp::Named("Phi", mod1["Phi"]),
                                        Rcpp::Named("Z", mod1["Z"]),
                                        Rcpp::Named("loglik", mod1["loglik"]));

  return mod2;
}

//' Tests BFOC function
//'
//' @name GetStuff
//' @export
// [[Rcpp::export]]
Rcpp::List TestEstimateInitialTT(const int tot_mcmc_iters, const int r_stored_iters,
                                 const double beta_N_t, const int N_t,
                                 const std::string directory){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
  arma::mat nu;
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++)
  {
    Phi.slice(i) = (3-i) * 0.1 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi(100, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  for(int i = 0; i < 100; i++){
    if(i < 10){
      Z.row(i) = {1, 0, 0};
    }else if(i < 20){
      Z.row(i) = {0, 1, 0};
    }else if(i < 30){
      Z.row(i) = {0, 0, 1};
    }

    while(arma::accu(Z.row(i)) == 0){
      Z.row(i) = arma::randi<arma::rowvec>(3, arma::distr_param(0,1));
    }
  }

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < nu.n_rows; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  //estimate B-spline expansion
  arma::mat theta = BasisExpansion(y_obs, B_obs, 100, 3, 8);
  //estimate Z matrix
  arma::mat Z_est = ZInitialState(B_obs, theta, 50, 3, 100, 0.001);
  // estimate sigma
  double sigma_est = SigmaInitialState(y_obs, B_obs, theta, 100);
  // estimate nu
  arma::mat nu_est = NuInitialState(B_obs, Z_est, theta, 100);
  //get rest of estimates
  Rcpp::List output = PhiChiInitialState(Z, y_obs, t_obs1, 100, 3, 8, 3,
                                         1000, 200, t_star1, 3, 0.7, 1, 2, 3,
                                         1, 1, sqrt(1), sqrt(1), 1, 1, 1, 1,
                                         nu, sigma_sq);

  // start MCMC sampling
  Rcpp::List mod1 = BFOC_U_TT(y_obs, t_obs1, n_funct, 3, 8, 3, tot_mcmc_iters,
                           r_stored_iters, t_star1, 3, 0.7, 1, 2, 3, 1, 1,
                           sqrt(1), sqrt(1), 1, 1, 1, 1, directory, Z,
                           output["A_est"], output["pi_est"], output["tau_est"],
                          output["delta_est"], nu, Phi,
                          output["gamma_est"], chi,
                          output["y_star_est"], beta_N_t, N_t, sigma_sq);

  Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
                                        Rcpp::Named("y_obs", y_obs),
                                        Rcpp::Named("nu_true", nu),
                                        Rcpp::Named("Phi_true", Phi),
                                        Rcpp::Named("nu", mod1["nu"]),
                                        Rcpp::Named("y_star", mod1["y_star"]),
                                        Rcpp::Named("chi", mod1["chi"]),
                                        Rcpp::Named("pi", mod1["pi"]),
                                        Rcpp::Named("A", mod1["A"]),
                                        Rcpp::Named("delta", mod1["delta"]),
                                        Rcpp::Named("sigma", mod1["sigma"]),
                                        Rcpp::Named("tau", mod1["tau"]),
                                        Rcpp::Named("gamma", mod1["gamma"]),
                                        Rcpp::Named("Phi", mod1["Phi"]),
                                        Rcpp::Named("Z", mod1["Z"]),
                                        Rcpp::Named("loglik", mod1["loglik"]));

  return mod2;
}

//' Tests BFOC function
//'
//' @name GetStuff
//' @export
// [[Rcpp::export]]
Rcpp::List TestEstimateInitialMTT(const int tot_mcmc_iters, const int r_stored_iters, const int n_temp_trans,
                                 const double beta_N_t, const int N_t,
                                 const std::string directory, const double sigma_sq){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
  arma::mat nu;
  nu.load("c:\\Projects\\BayesFPMM\\data\\nu.txt");


  // Make Phi matrix
  arma::cube Phi;
  Phi.load("c:\\Projects\\BayesFPMM\\data\\Phi.txt");
  // double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi;
  chi.load("c:\\Projects\\BayesFPMM\\data\\chi.txt");


  // Make Z matrix
  arma::mat Z;
  Z.load("c:\\Projects\\BayesFPMM\\data\\Z.txt");

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < nu.n_rows; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  // estimate B-spline expansion
    arma::mat theta = BasisExpansion(y_obs, B_obs, 100, 3, 8);
  //estimate Z matrix
  arma::mat Z_est = ZInitialState(B_obs, theta, 50, 3, 100, 0.001);
  // estimate sigma
  double sigma_est = SigmaInitialState(y_obs, B_obs, theta, 100);
  // estimate nu
  arma::mat nu_est = NuInitialState(B_obs, Z_est, theta, 100);
  // get rest of estimates
  Rcpp::List output = PhiChiInitialState(Z_est, y_obs, t_obs1, 100, 3, 8, 3,
                                         1000, 200, t_star1, 3, 0.7, 1, 2, 3,
                                         1, 1, sqrt(1), sqrt(1), 1, 1, 1, 1,
                                         nu_est, sigma_est);

  // start MCMC sampling
  Rcpp::List mod1 = BFOC_U_MTT(y_obs, t_obs1, n_funct, 3, 8, 3, tot_mcmc_iters,
                               r_stored_iters, n_temp_trans, t_star1, 3, 0.7, 1, 2, 3, 1, 1,
                               sqrt(1), sqrt(1), 1, 1, 1, 1, directory, Z_est,
                               output["A_est"], output["pi_est"], output["tau_est"],
                               output["delta_est"], nu_est, output["Phi_est"],
                               output["gamma_est"], output["chi_est"],
                               output["y_star_est"], beta_N_t, N_t, sigma_est);

  Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
                                        Rcpp::Named("y_obs", y_obs),
                                        Rcpp::Named("nu_true", nu),
                                        Rcpp::Named("Phi_true", Phi),
                                        Rcpp::Named("nu", mod1["nu"]),
                                        Rcpp::Named("y_star", mod1["y_star"]),
                                        Rcpp::Named("chi", mod1["chi"]),
                                        Rcpp::Named("pi", mod1["pi"]),
                                        Rcpp::Named("A", mod1["A"]),
                                        Rcpp::Named("delta", mod1["delta"]),
                                        Rcpp::Named("sigma", mod1["sigma"]),
                                        Rcpp::Named("tau", mod1["tau"]),
                                        Rcpp::Named("gamma", mod1["gamma"]),
                                        Rcpp::Named("Phi", mod1["Phi"]),
                                        Rcpp::Named("Z", mod1["Z"]),
                                        Rcpp::Named("loglik", mod1["loglik"]));

  return mod2;
}

//' Tests BFOC function
//'
//' @name GetStuff
//' @export
// [[Rcpp::export]]
Rcpp::List TestEstimateInitialTempladder(const double beta_N_t, const int N_t){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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

  arma::mat nu;
  nu.load("c:\\Projects\\BayesFPMM\\data\\nu.txt");


  // Make Phi matrix
  arma::cube Phi;
  Phi.load("c:\\Projects\\BayesFPMM\\data\\Phi.txt");
  double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi;
  chi.load("c:\\Projects\\BayesFPMM\\data\\chi.txt");


  // Make Z matrix
  arma::mat Z;
  Z.load("c:\\Projects\\BayesFPMM\\data\\Z.txt");

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < nu.n_rows; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  // estimate B-spline expansion
  // arma::mat theta = BasisExpansion(y_obs, B_obs, 100, 3, 8);
  // //estimate Z matrix
  // arma::mat Z_est = ZInitialState(B_obs, theta, 50, 3, 100, 0.001);
  // // estimate sigma
  // double sigma_est = SigmaInitialState(y_obs, B_obs, theta, 100);
  // // estimate nu
  // arma::mat nu_est = NuInitialState(B_obs, Z_est, theta, 100);
  // get rest of estimates
  Rcpp::List output = PhiChiInitialState(Z, y_obs, t_obs1, 100, 3, 8, 3,
                                         1000, 200, t_star1, 3, 0.7, 1, 2, 3,
                                         1, 1, sqrt(1), sqrt(1), 1, 1, 1, 1,
                                         nu, sigma_sq);

  // start MCMC sampling
  Rcpp::List mod1 = BFOC_U_Templadder(y_obs, t_obs1, n_funct, 3, 8, 3, 10, 10, t_star1, 3, 0.7, 1, 2, 3, 1, 1,
                              sqrt(1), sqrt(1), 1, 1, 1, 1, Z,
                              output["A_est"], output["pi_est"], output["tau_est"],
                                     output["delta_est"], nu, Phi,
                                     output["gamma_est"], chi,
                                     output["y_star_est"], beta_N_t, N_t, sigma_sq);

  Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
                                        Rcpp::Named("y_obs", y_obs),
                                        Rcpp::Named("nu_true", nu),
                                        Rcpp::Named("Phi_true", Phi),
                                        Rcpp::Named("nu", mod1["nu"]),
                                        Rcpp::Named("y_star", mod1["y_star"]),
                                        Rcpp::Named("chi", mod1["chi"]),
                                        Rcpp::Named("pi", mod1["pi"]),
                                        Rcpp::Named("A", mod1["A"]),
                                        Rcpp::Named("delta", mod1["delta"]),
                                        Rcpp::Named("sigma", mod1["sigma"]),
                                        Rcpp::Named("tau", mod1["tau"]),
                                        Rcpp::Named("gamma", mod1["gamma"]),
                                        Rcpp::Named("Phi", mod1["Phi"]),
                                        Rcpp::Named("Z", mod1["Z"]));

  return mod2;
}

//' Tests BFOC function
//'
//' @name GetStuff
//' @export
// [[Rcpp::export]]
Rcpp::List TestEstimateInitialTemp(const int tot_mcmc_iters, const int r_stored_iters,
                                   const double temp, const std::string directory){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
  arma::mat nu;
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++)
  {
    Phi.slice(i) = (3-i) * 0.1 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi(100, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  for(int i = 0; i < 100; i++){
    if(i < 10){
      Z.row(i) = {1, 0, 0};
    }else if(i < 20){
      Z.row(i) = {0, 1, 0};
    }else if(i < 30){
      Z.row(i) = {0, 0, 1};
    }

    while(arma::accu(Z.row(i)) == 0){
      Z.row(i) = arma::randi<arma::rowvec>(3, arma::distr_param(0,1));
    }
  }

  arma::field<arma::vec> y_obs(100, 1);
  arma::field<arma::mat> y_star(100, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 100; j++){
    mean = arma::zeros(8);
    for(int l = 0; l < nu.n_rows; l++){
      mean = mean + Z(j,l) * nu.row(l).t();
      for(int m = 0; m < Phi.n_slices; m++){
        mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
      }
    }
    y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
      arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
  }

  // estimate B-spline expansion
  arma::mat theta = BasisExpansion(y_obs, B_obs, 100, 3, 8);
  //estimate Z matrix
  arma::mat Z_est = ZInitialState(B_obs, theta, 50, 3, 100, 0.001);
  // estimate sigma
  double sigma_est = SigmaInitialState(y_obs, B_obs, theta, 100);
  // estimate nu
  arma::mat nu_est = NuInitialState(B_obs, Z_est, theta, 100);
  // get rest of estimates
  Rcpp::List output = PhiChiInitialState(Z_est, y_obs, t_obs1, 100, 3, 8, 3,
                                         1000, 200, t_star1, 3, 0.7, 1, 2, 3,
                                         1, 1, sqrt(1), sqrt(1), 1, 1, 1, 1,
                                         nu_est, sigma_est);

  // start MCMC sampling
  Rcpp::List mod1 = BFOC_U_Temp(y_obs, t_obs1, n_funct, 3, 8, 3, tot_mcmc_iters,
                           r_stored_iters, t_star1, 3, 0.7, 1, 2, 3, 1, 1,
                           sqrt(1), sqrt(1), 1, 1, 1, 1, directory, Z_est,
                           output["A_est"], output["pi_est"], output["tau_est"],
                           output["delta_est"], nu_est, output["Phi_est"],
                           output["gamma_est"], output["chi_est"],
                           output["y_star_est"], temp, sigma_est);

  Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
                                        Rcpp::Named("y_obs", y_obs),
                                        Rcpp::Named("nu_true", nu),
                                        Rcpp::Named("Phi_true", Phi),
                                        Rcpp::Named("nu", mod1["nu"]),
                                        Rcpp::Named("y_star", mod1["y_star"]),
                                        Rcpp::Named("chi", mod1["chi"]),
                                        Rcpp::Named("pi", mod1["pi"]),
                                        Rcpp::Named("A", mod1["A"]),
                                        Rcpp::Named("delta", mod1["delta"]),
                                        Rcpp::Named("sigma", mod1["sigma"]),
                                        Rcpp::Named("tau", mod1["tau"]),
                                        Rcpp::Named("gamma", mod1["gamma"]),
                                        Rcpp::Named("Phi", mod1["Phi"]),
                                        Rcpp::Named("Z", mod1["Z"]),
                                        Rcpp::Named("loglik", mod1["loglik"]));

  return mod2;
}
//' Tests updating Z
//'
//' @name TestUpdateZ
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateZTempered(const double beta)
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
    updateZTempered(beta, y_obs, y_star, B_obs, B_star, Phi, nu, chi, pi,
            sigma_sq, 0.6, i, 100, Z_ph, Z_samp);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
                                      Rcpp::Named("Z",Z),
                                      Rcpp::Named("f_obs", y_obs),
                                      Rcpp::Named("f_star", y_star));
  return mod;
}

//' Tests updating Phi using temperature
//'
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdatePhiTempered(const double beta)
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
    updatePhiTempered(beta, y_obs, y_star, B_obs, B_star, nu, gamma, tilde_tau, Z, chi,
              sigma_sq, i, 100, m_1, M_1, Phi_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Phi", Phi),
                                      Rcpp::Named("Phi_samp", Phi_samp));
  return mod;
}

//' Tests updating Nu
//'
//' @name TestUpdateNuTemperd
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateNuTempered(const double beta){
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
  arma::vec log_lik = arma::zeros(100);
  for(int i = 0; i < 100; i++){
    updateNuTempered(beta, y_obs, y_star, B_obs, B_star, tau, Phi, Z, chi, sigma_sq, i, 100,
             P, b_1, B_1, Nu_samp);
    log_lik(i) = calcLikelihood(y_obs, y_star, B_obs, B_star, Nu_samp.slice(i),
            Phi, Z, chi,i, sigma_sq);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu_samp", Nu_samp),
                                      Rcpp::Named("nu", nu),
                                      Rcpp::Named("log_lik", log_lik));
  return mod;
}

//' Tests updating Sigma
//'
//' @name TestUpdateSigmaTempered
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateSigmaTempered(const double beta){
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
    updateSigmaTempered(beta, y_obs, y_star, B_obs, B_star, alpha_0, beta_0, nu, Phi, Z, chi,
                i, 100, sigma_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("sigma_samp", sigma_samp),
                                      Rcpp::Named("sigma", sigma_sq));
  return mod;
}

//' Tests updating chi
//'
//' @name TestUpdateChiTempered
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateChiTempered(const double beta){
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



  arma::cube chi_samp(100, 5, 10000, arma::fill::zeros);
  chi_samp.slice(0) = chi;
  for(int i = 0; i < 10000; i++){
    for(int j = 0; j < 100; j++){
      y_star(j,0) = arma::zeros(10000, B_star(j,0).n_rows);
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
      for(int k = 1; k < 10000; k++){
        y_star(j, 0).row(k) = y_star(j, 0).row(0);
      }
    }
    updateChiTempered(beta, y_obs, y_star, B_obs, B_star, Phi, nu, Z, sigma_sq, i, 10000,
              chi_samp);
  }
  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("chi_samp", chi_samp),
                                      Rcpp::Named("chi", chi),
                                      Rcpp::Named("Z", Z));
  return mod;
}

//' Tests updating YstarTempered
//'
//' @name TestUpdateYStarTempered
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateYStarTempered(const double beta){
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

  arma::field<arma::mat> y_star_samp(100, 1);

  for(int i = 0; i < 100; i++){
    y_star_samp(i,0) = arma::zeros(1000, B_star(i,0).n_rows);
  }

  for(int i = 0; i < 1000; i++){
    updateYStarTempered(beta, B_star, nu, Phi, Z, chi, sigma_sq, i, 1000, y_star_samp);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("y_star_samp", y_star_samp),
                                      Rcpp::Named("y_star", y_star),
                                      Rcpp::Named("y_obs", y_obs));
  return mod;
}

//' simulates parameters
//'
//' @name getparams
//' @export
// [[Rcpp::export]]
void getparms(){
  arma::field<arma::vec> t_obs1(100,1);
  arma::field<arma::vec> t_star1(100,1);
  int n_funct = 100;
  for(int i = 0; i < n_funct; i++){
    t_obs1(i,0) =  arma::regspace(0, 10, 990);
    t_star1(i,0) = arma::regspace(0, 50, 950);
  }

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
  arma::mat bspline_mat {bspline.basis(true)};
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
  arma::mat nu(3,8, arma::fill::randn);
  nu = 3 * nu;

  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++)
  {
    Phi.slice(i) = (3-i) * 0.5 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.005;

  // Make chi matrix
  arma::mat chi(100, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
  for(int i = 0; i < 100; i++){
    if(i < 10){
      Z.row(i) = {1, 0, 0};
    }else if(i < 20){
      Z.row(i) = {0, 1, 0};
    }else if(i < 30){
      Z.row(i) = {0, 0, 1};
    }

    while(arma::accu(Z.row(i)) == 0){
      Z.row(i) = arma::randi<arma::rowvec>(3, arma::distr_param(0,1));
    }
  }

  //save parameters
  nu.save("c:\\Projects\\BayesFPMM\\data\\nu.txt", arma::arma_ascii);
  chi.save("c:\\Projects\\BayesFPMM\\data\\chi.txt", arma::arma_ascii);
  Phi.save("c:\\Projects\\BayesFPMM\\data\\Phi.txt", arma::arma_ascii);
  Z.save("c:\\Projects\\BayesFPMM\\data\\Z.txt", arma::arma_ascii);

}

//' Tests updating Z using partial membership model
//'
//' @name TestUpdateZ_PM
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdateZ_PM(){
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
    Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
  }
  double sigma_sq = 0.001;

  // Make chi matrix
  arma::mat chi(100, 5, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(100, 3);
  arma::mat alpha(100,3, arma::fill::randu);
  alpha = alpha * 100;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = rdirichlet(alpha.row(i).t()).t();
  }

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
  arma::vec pi = {1, 1, 1};

  // Initialize placeholder
  arma::vec Z_ph = arma::zeros(3);
  arma::vec Z_tilde_ph = arma::zeros(3);


  //Initialize Z_samp
  arma::cube Z_samp = arma::ones(100, 3, 1000);
  arma::cube Z_tilde = arma::zeros(100, 3, 1000);
  for(int i = 0; i < 100; i++){
    Z_samp.slice(0).row(i) = rdirichlet(pi).t();
    Z_tilde.slice(0).row(i) = Z_samp.slice(0).row(i);
  }
  for(int i = 0; i < 1000; i++)
  {
    updateZ_PM(y_obs, y_star, B_obs, B_star, Phi, nu, chi, pi,
            sigma_sq, i, 1000, 1.0, 0.1, Z_tilde, Z_tilde_ph, Z_ph, Z_samp);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
                                      Rcpp::Named("Z",Z),
                                      Rcpp::Named("Z_tilde", Z_tilde),
                                      Rcpp::Named("f_obs", y_obs),
                                      Rcpp::Named("f_star", y_star));
  return mod;

}

//' Tests updating pi using partial membership model
//'
//' @name TestUpdateZ_PM
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdatepi_PM(){

  // Make Z matrix
  arma::mat Z(100, 3);
  arma::vec c(3, arma::fill::randu);
  arma::vec pi = rdirichlet(c);

  // setting alpha_3 = 100
  arma:: vec alpha = pi * 100;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = rdirichlet(alpha).t();
  }

  // Initialize placeholder
  arma::vec pi_ph = arma::zeros(3);
  arma::vec pi_tilde_ph = arma::zeros(3);


  //Initialize Z_samp
  arma::mat pi_samp = arma::ones(3, 1000);
  arma::mat pi_tilde = arma::zeros(3, 1000);
  pi_samp.col(0) = rdirichlet(c);
  pi_tilde.col(0) =  pi_samp.col(0);

  for(int i = 0; i < 1000; i++)
  {
    updatePi_PM(100 ,Z, c, i, 1000, 0.1, pi_tilde, pi_ph, pi_tilde_ph, pi_samp);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("pi_samp", pi_samp),
                                      Rcpp::Named("pi",pi),
                                      Rcpp::Named("pi_tilde", pi_tilde),
                                      Rcpp::Named("Z", Z));
  return mod;
}

//' Tests updating pi using partial membership model
//'
//' @name TestUpdateZ_PM
//' @export
// [[Rcpp::export]]
Rcpp::List TestUpdatealpha3_PM(){

  // Make Z matrix
  arma::mat Z(100, 3);
  arma::vec c(3, arma::fill::randu);
  arma::vec pi = rdirichlet(c);

  // setting alpha_3 = 10
  arma:: vec alpha = pi * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = rdirichlet(alpha).t();
  }

  arma::vec alpha_3(1000, arma::fill::ones);

  for(int i = 0; i < 1000; i++)
  {
    updateAlpha3(pi, 0.5, Z, i, 1000, 0.1, alpha_3);
  }

  Rcpp::List mod = Rcpp::List::create(Rcpp::Named("alpha3_samp", alpha_3));
  return mod;
}


