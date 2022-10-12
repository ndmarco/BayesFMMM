#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFMMM.h>

// Tests updating Phi
//
arma::field<arma::cube> TestUpdatePhi(){
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

  arma::field<arma::vec> y_obs(40, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
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
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::mat tilde_tau = {{1, 2}, {1, 2}, {1, 2}};
  for(int i = 0; i < 250; i++){
    BayesFMMM::updatePhi(y_obs, B_obs, nu, gamma, tilde_tau, Z, chi,
                         sigma_sq, i, 250, m_1, M_1, Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

// Tests updating Phi
//
arma::field<arma::cube> TestUpdatePhiTempered(){
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

  arma::field<arma::vec> y_obs(40, 1);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
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
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::mat tilde_tau = {{1, 2}, {1, 2}, {1, 2}};
  for(int i = 0; i < 250; i++){
    BayesFMMM::updatePhiTempered(0.5, y_obs, B_obs, nu, gamma, tilde_tau, Z, chi,
                                 sigma_sq, i, 250, m_1, M_1, Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

// Tests updating Phi multivariate
//
arma::field<arma::cube> TestUpdatePhiMV(){
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

  arma::mat y_obs = arma::zeros(40, 8);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
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
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::mat tilde_tau = {{1, 2}, {1, 2}, {1, 2}};
  for(int i = 0; i < 250; i++){
    BayesFMMM::updatePhiMV(y_obs, nu, gamma, tilde_tau, Z, chi,
                           sigma_sq, i, 250, m_1, M_1, Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

// Tests updating Phi multivariate using a tempered likelihood
//
arma::field<arma::cube> TestUpdatePhiTemperedMV(){
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

  arma::mat y_obs = arma::zeros(40, 8);
  arma::vec mean = arma::zeros(8);

  for(int j = 0; j < 40; j++){
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
  arma::vec pi = {0.5, 0.5, 0.5};

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::mat tilde_tau = {{1, 2}, {1, 2}, {1, 2}};
  for(int i = 0; i < 250; i++){
    BayesFMMM::updatePhiTemperedMV(0.5, y_obs, nu, gamma, tilde_tau, Z, chi,
                           sigma_sq, i, 250, m_1, M_1, Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

// Tests updating Gamma
//
arma::field<arma::cube> TestUpdateGamma(){
  // Specify hyperparameters
  double nu = 1;
  // Make Delta vector
  arma::mat Delta = {{2,3}, {2,3}, {2,3}};
  // Make Gamma cube
  arma::cube Gamma(3,8,2);
  for(int i=0; i < 2; i++){
    for(int j = 0; j < 3; j++){
      for(int k = 0; k < 8; k++){
        Gamma(j,k,i) =  R::rgamma(nu/2, 2/nu);
      }
    }
  }

  // Make Phi matrix
  arma::cube Phi(3,8,2);
  arma::field<arma::cube> gamma(10000,1);
  for(int i = 0; i < 10000; i++){
    gamma(i,0) = arma::zeros(3,8,2);
  }
  for(int m = 0; m < 10000; m++){
    for(int j=0; j < 3; j++){
      double tau  = 1;
      for(int i=0; i < 2; i++){
        tau = tau * Delta(j, i);
        for(int k=0; k < 8; k++){
          Phi(j,k,i) = R::rnorm(0, (1/ std::pow(Gamma(j,k,i)*tau, 0.5)));
        }
      }
    }
    BayesFMMM::updateGamma(0.001, Delta, Phi, m, 10000, gamma);
  }
  arma::vec gamma_ph = arma::zeros(10000);
  arma::cube gamma_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 0; k < 10000; k++){
          gamma_ph(k) = gamma(k,0)(i,j,l);
        }
        gamma_est(i,j,l) = arma::median(gamma_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = gamma_est;
  mod(1,0) = Gamma;
  return mod;
}

// Tests updating Delta
//
arma::field<arma::mat> TestUpdateDelta(){
  // Specify hyperparameters
  arma::mat a_12 = {{2, 2}, {2, 2}, {2, 2}};
  // Make Delta vector
  arma::mat Delta = arma::zeros(3, 5);
  for(int j = 0; j < 3; j++){
    for(int i=0; i < 5; i++){
      Delta(j, i) = R::rgamma(2, 1);
    }
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
  arma::cube delta = arma::ones(3, 5, 10000);
  for(int m = 0; m < 10000; m++){
    for(int j=0; j < 3; j++){
      double tau  = 1;
      for(int i=0; i < 5; i++){
        tau = tau * Delta(j, i);
        for(int k=0; k < 8; k++){
          Phi(j,k,i) = R::rnorm(0, (1/ std::pow(Gamma(j,k,i)*tau, 0.5)));
        }
      }
    }
    BayesFMMM::updateDelta(Phi, Gamma, a_12, m, 10000, delta);
  }
  arma::mat Delta_est = arma::zeros(3, 5);
  arma::vec ph = arma::zeros(10000);
  for(int j = 0; j < 3; j++){
    for(int i = 0; i < 5; i++){
      for(int m = 0; m < 10000; m++){
        ph(m) = delta(j, i, m);
      }
      Delta_est(j, i) = arma::median(ph);
    }
  }

  arma::field<arma::mat> mod (2,1);
  mod(0,0) = Delta_est;
  mod(1,0) = Delta;
  return mod;
}

// Tests updating A
//
arma::field<arma::mat> TestUpdateA(){
  double a_1 = 2;
  double a_2 = 3;
  arma::mat delta = arma::zeros(3, 5);
  double alpha1 = 2;
  double beta1 = 1;
  double alpha2 = 3;
  double beta2 = 1;
  arma::cube A = arma::ones(3, 2, 1000);
  for(int i = 0; i < 1000; i++){
    for(int l = 0; l < 3; l++){
      for(int j = 0; j < 5; j++){
        if(j == 0){
          delta(l,j) = 2;
        }else{
          delta(l,j) = 3;
        }
      }
    }
    BayesFMMM::updateA(alpha1, beta1, alpha2, beta2, delta, 1, 1, i, 1000, A);
  }
  arma::mat A_est = arma::zeros(3, 2);
  arma::vec ph = arma::zeros(1000);
  for(int j = 0; j < 3; j++){
    for(int i = 0; i < 2; i++){
      for(int m =0; m < 1000; m++){
        ph(m) = A(j, i, m);
      }
      A_est(j, i) = arma::median(ph);
    }
  }
  arma::field<arma::mat> mod(2,1);
  mod(0,0) = A_est;
  mod(1,0) = {{2,3}, {2,3}, {2,3}};

  return mod;
}

////////////////////////
// Covariate Adjusted //
////////////////////////

// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdatePhiCovariateAdj(){
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

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::mat tilde_tau = {{1, 2}, {1, 2}, {1, 2}};
  for(int i = 0; i < 250; i++){
    BayesFMMM::updatePhiCovariateAdj(y_obs, B_obs, nu, eta, gamma, tilde_tau, xi,
                                     Z, chi, sigma_sq, X, i, 250, m_1, M_1,
                                     Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdatePhiTemperedCovariateAdj(){
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

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::mat tilde_tau = {{1, 2}, {1, 2}, {1, 2}};
  for(int i = 0; i < 250; i++){
    BayesFMMM::updatePhiTemperedCovariateAdj(0.5, y_obs, B_obs, nu, eta, gamma,
                                             tilde_tau, xi, Z, chi, sigma_sq, X,
                                             i, 250, m_1, M_1, Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

// Tests updating Phi using a covariate adjusted multivariate model
//
arma::field<arma::cube> TestUpdatePhiMVCovariateAdj(){
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

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::mat tilde_tau = {{1, 2}, {1, 2}, {1, 2}};
  for(int i = 0; i < 250; i++){
    BayesFMMM::updatePhiMVCovariateAdj(y_obs, nu, eta, gamma, tilde_tau, xi, Z,
                                       chi, sigma_sq, X, i, 250, m_1, M_1,
                                       Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

// Tests updating Phi using a tempered likelihood for the covariate adjusted multivariate model
//
arma::field<arma::cube> TestUpdatePhiTemperedMVCovariateAdj(){
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

  arma::field<arma::cube> Phi_samp(250, 1);
  for(int i = 0; i < 250 ; i++){
    Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
  }
  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
  gamma = gamma * 10;
  arma::mat tilde_tau = {{1, 2}, {1, 2}, {1, 2}};
  for(int i = 0; i < 250; i++){
    BayesFMMM::updatePhiTemperedMVCovariateAdj(0.5, y_obs, nu, eta, gamma,
                                               tilde_tau, xi, Z, chi, sigma_sq,
                                               X, i, 250, m_1, M_1, Phi_samp);
  }

  arma::vec phi_ph = arma::zeros(150);
  arma::cube phi_est = arma::zeros(3,8,2);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int k = 100; k < 250; k++){
          phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
        }
        phi_est(i,j,l) = arma::median(phi_ph);
      }
    }
  }
  arma::field<arma::cube> mod (2,1);
  mod(0,0) = phi_est;
  mod(1,0) = Phi;
  return mod;
}

context("Unit tests for Phi parameters") {
  test_that("Sampler for Phi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhi();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Tempered sampler for Phi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhiTempered();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Phi parameters in multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhiMV();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Tempered sampler for Phi parameters in multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhiTemperedMV();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Gamma parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateGamma();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.5){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Delta parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateDelta();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int j = 0; j < est.n_rows; j++){
      for(int i = 0; i < est.n_cols; i++){
        if(std::abs(est(j,i) - truth(j,i)) > 2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for A parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateA();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int j = 0; j < est.n_rows; j++){
      for(int i = 0; i < est.n_cols; i++){
        if(std::abs(est(j,i) - truth(j,i)) > 0.5){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Phi parameters under the ßcovariate adjusted model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhiCovariateAdj();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Tempered sampler for Phi parameters under the covariate adjusted model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhiTemperedCovariateAdj();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Phi parameters in a covariate adjusted multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhiMVCovariateAdj();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Tempered sampler for Phi parameters in the covariate adjusted multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdatePhiTemperedMVCovariateAdj();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.3){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }
}

