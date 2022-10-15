#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFMMM.h>

// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdateXiCovariateAdj(){
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

  arma::field<arma::cube> xi_samp(250,3);
  for(int i = 0; i < 3; i++){
    xi_samp(0,i) = arma::zeros(8,2,2);
  }
  for(int i = 1; i < 250; i++){
    xi_samp(i,0) = xi_samp(0,0);
    xi_samp(i,1) = xi_samp(0,1);
    xi_samp(i,2) = xi_samp(0,2);
  }

  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::field<arma::cube> gamma_xi (250, 3);
  for(int i = 0; i < 3; i++){
    gamma_xi(0,i) = arma::ones(xi(0,0).n_rows, xi(0,0).n_cols, xi(0,0).n_slices)* 10;
  }
  for(int i = 1; i < 250; i++){
    gamma_xi(i,0) = gamma_xi(0,0);
    gamma_xi(i,1) = gamma_xi(0,1);
    gamma_xi(i,2) = gamma_xi(0,2);
  }

  arma::cube tilde_tau_gamma = arma::ones(3,2,2);
  for(int i = 0; i < 250; i++){
    BayesFMMM::updateXiCovariateAdj(y_obs, B_obs, nu, eta, gamma_xi, tilde_tau_gamma, Phi,
                                     Z, chi, sigma_sq, X, i, 250, m_1, M_1,
                                     xi_samp);
  }

  arma::vec xi_ph = arma::zeros(150);
  arma::field<arma::cube> xi_est(1,3);
  for(int i = 0; i < 3; i++){
    xi_est(0,i) = arma::zeros(8,2,2);
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int d = 0; d < 2; d++){
          for(int k = 100; k < 250; k++){
            xi_ph(k - 100) = xi_samp(k,i)(j,l,d);
          }
          xi_est(0,i)(j,l,d) = arma::median(xi_ph);
        }
      }
    }
  }
  arma::field<arma::cube> mod (2,3);
  for(int i = 0; i < 3; i++){
    mod(0,i) = xi(0,i);
    mod(1,i) = xi_est(0,i);
  }
  return mod;
}


// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdateXiTemperedCovariateAdj(){
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

  arma::field<arma::cube> xi_samp(250,3);
  for(int i = 0; i < 3; i++){
    xi_samp(0,i) = arma::zeros(8,2,2);
  }
  for(int i = 1; i < 250; i++){
    xi_samp(i,0) = xi_samp(0,0);
    xi_samp(i,1) = xi_samp(0,1);
    xi_samp(i,2) = xi_samp(0,2);
  }

  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::field<arma::cube> gamma_xi (250, 3);
  for(int i = 0; i < 3; i++){
    gamma_xi(0,i) = arma::ones(xi(0,0).n_rows, xi(0,0).n_cols, xi(0,0).n_slices)* 10;
  }
  for(int i = 1; i < 250; i++){
    gamma_xi(i,0) = gamma_xi(0,0);
    gamma_xi(i,1) = gamma_xi(0,1);
    gamma_xi(i,2) = gamma_xi(0,2);
  }

  arma::cube tilde_tau_gamma = arma::ones(3,2,2);
  for(int i = 0; i < 250; i++){
    BayesFMMM::updateXiTemperedCovariateAdj(0.5, y_obs, B_obs, nu, eta, gamma_xi,
                                            tilde_tau_gamma, Phi, Z, chi,
                                            sigma_sq, X, i, 250, m_1, M_1, xi_samp);
  }

  arma::vec xi_ph = arma::zeros(150);
  arma::field<arma::cube> xi_est(1,3);
  for(int i = 0; i < 3; i++){
    xi_est(0,i) = arma::zeros(8,2,2);
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int d = 0; d < 2; d++){
          for(int k = 100; k < 250; k++){
            xi_ph(k - 100) = xi_samp(k,i)(j,l,d);
          }
          xi_est(0,i)(j,l,d) = arma::median(xi_ph);
        }
      }
    }
  }
  arma::field<arma::cube> mod (2,3);
  for(int i = 0; i < 3; i++){
    mod(0,i) = xi(0,i);
    mod(1,i) = xi_est(0,i);
  }
  return mod;
}


// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdateXiMVCovariateAdj(){

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

  arma::field<arma::cube> xi_samp(250,3);
  for(int i = 0; i < 3; i++){
    xi_samp(0,i) = arma::zeros(8,2,2);
  }
  for(int i = 1; i < 250; i++){
    xi_samp(i,0) = xi_samp(0,0);
    xi_samp(i,1) = xi_samp(0,1);
    xi_samp(i,2) = xi_samp(0,2);
  }

  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::field<arma::cube> gamma_xi (250, 3);
  for(int i = 0; i < 3; i++){
    gamma_xi(0,i) = arma::ones(xi(0,0).n_rows, xi(0,0).n_cols, xi(0,0).n_slices)* 10;
  }
  for(int i = 1; i < 250; i++){
    gamma_xi(i,0) = gamma_xi(0,0);
    gamma_xi(i,1) = gamma_xi(0,1);
    gamma_xi(i,2) = gamma_xi(0,2);
  }

  arma::cube tilde_tau_gamma = arma::ones(3,2,2);
  for(int i = 0; i < 250; i++){
    BayesFMMM::updateXiMVCovariateAdj(y_obs, nu, eta, gamma_xi,
                                      tilde_tau_gamma, Phi, Z, chi,
                                      sigma_sq, X, i, 250, m_1, M_1, xi_samp);
  }

  arma::vec xi_ph = arma::zeros(150);
  arma::field<arma::cube> xi_est(1,3);
  for(int i = 0; i < 3; i++){
    xi_est(0,i) = arma::zeros(8,2,2);
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int d = 0; d < 2; d++){
          for(int k = 100; k < 250; k++){
            xi_ph(k - 100) = xi_samp(k,i)(j,l,d);
          }
          xi_est(0,i)(j,l,d) = arma::median(xi_ph);
        }
      }
    }
  }
  arma::field<arma::cube> mod (2,3);
  for(int i = 0; i < 3; i++){
    mod(0,i) = xi(0,i);
    mod(1,i) = xi_est(0,i);
  }
  return mod;
}

// Tests updating Phi under the covariate adjusted model
//
arma::field<arma::cube> TestUpdateXiTemperedMVCovariateAdj(){

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

  arma::field<arma::cube> xi_samp(250,3);
  for(int i = 0; i < 3; i++){
    xi_samp(0,i) = arma::zeros(8,2,2);
  }
  for(int i = 1; i < 250; i++){
    xi_samp(i,0) = xi_samp(0,0);
    xi_samp(i,1) = xi_samp(0,1);
    xi_samp(i,2) = xi_samp(0,2);
  }

  arma::vec m_1(Phi.n_cols, arma::fill::zeros);
  arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
  arma::field<arma::cube> gamma_xi (250, 3);
  for(int i = 0; i < 3; i++){
    gamma_xi(0,i) = arma::ones(xi(0,0).n_rows, xi(0,0).n_cols, xi(0,0).n_slices)* 10;
  }
  for(int i = 1; i < 250; i++){
    gamma_xi(i,0) = gamma_xi(0,0);
    gamma_xi(i,1) = gamma_xi(0,1);
    gamma_xi(i,2) = gamma_xi(0,2);
  }

  arma::cube tilde_tau_gamma = arma::ones(3,2,2);
  for(int i = 0; i < 250; i++){
    BayesFMMM::updateXiTemperedMVCovariateAdj(0.5, y_obs, nu, eta, gamma_xi,
                                              tilde_tau_gamma, Phi, Z, chi,
                                              sigma_sq, X, i, 250, m_1, M_1, xi_samp);
  }

  arma::vec xi_ph = arma::zeros(150);
  arma::field<arma::cube> xi_est(1,3);
  for(int i = 0; i < 3; i++){
    xi_est(0,i) = arma::zeros(8,2,2);
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int d = 0; d < 2; d++){
          for(int k = 100; k < 250; k++){
            xi_ph(k - 100) = xi_samp(k,i)(j,l,d);
          }
          xi_est(0,i)(j,l,d) = arma::median(xi_ph);
        }
      }
    }
  }
  arma::field<arma::cube> mod (2,3);
  for(int i = 0; i < 3; i++){
    mod(0,i) = xi(0,i);
    mod(1,i) = xi_est(0,i);
  }
  return mod;
}

// Tests updating Gamma
//
arma::field<arma::cube> TestUpdateGammaXi(){
  // Specify hyperparameters
  double nu = 1;
  // Make Delta vector
  arma::mat Delta_i = {{2,3}, {2,3}, {2,3}};
  arma::cube Delta = arma::zeros(3,2,2);
  Delta.slice(0) = Delta_i;
  Delta.slice(1) = Delta_i;
  // Make Gamma cube
  arma::field<arma::cube> Gamma(1, 3);
  for(int l=0; l < 3; l++){
    Gamma(0,l) = arma::zeros(8,2,2);
    for(int i=0; i < 2; i++){
      for(int j = 0; j < 2; j++){
        for(int k = 0; k < 8; k++){
          Gamma(0,l)(k,j,i) =  R::rgamma(nu/2, 2/nu);
        }
      }
    }
  }


  // Make Phi matrix
  arma::field<arma::cube> xi(10000,3);
  arma::field<arma::cube> gamma(10000,3);
  for(int l = 0; l < 3; l++){
    for(int i = 0; i < 10000; i++){
      gamma(i,l) = arma::zeros(8,2,2);
      xi(i,l) = arma::zeros(8,2,2);
    }
  }

  for(int m = 0; m < 10000; m++){
    for(int l = 0; l < 3; l++){
      for(int j = 0; j < 2; j++){
        double tau  = 1;
        for(int i=0; i < 2; i++){
          tau = tau * Delta(l, i, j);
          for(int k=0; k < 8; k++){
            xi(m,l)(k,j,i) = R::rnorm(0, (1/ std::pow(Gamma(0,l)(k,j,i)*tau, 0.5)));
          }
        }
      }
    }

    BayesFMMM::updateGammaXi(0.0001, Delta, xi, m, 10000, gamma);
  }
  arma::vec gamma_ph = arma::zeros(10000);
  arma::field<arma::cube> gamma_xi(1,3);
  for(int i = 0; i < 3; i++){
    gamma_xi(0,i) = arma::zeros(8,2,2);
    for(int j = 0; j < 8; j++){
      for(int l = 0; l < 2; l++){
        for(int d = 0; d < 2; d++){
          for(int k = 0; k < 10000; k++){
            gamma_ph(k) = gamma(k,i)(j,l,d);
          }
          gamma_xi(0,i)(j,l,d) = arma::median(gamma_ph);
        }
      }
    }
  }

  arma::field<arma::cube> mod (2,3);
  for(int i = 0; i < 3; i++){
    mod(0,i) = Gamma(0,i);
    mod(1,i) = gamma_xi(0,i);
  }

  return mod;
}

// Tests updating Delta
//
arma::field<arma::cube> TestUpdateDeltaXi(){
  // Specify hyperparameters
  arma::cube A_xi = 2 * arma::ones(3,2,2);

  // Make Delta vector
  arma::cube Delta = arma::zeros(3, 5, 2);
  for(int l = 0; l < 2; l++){
    for(int j = 0; j < 3; j++){
      for(int i = 0; i < 5; i++){
        Delta(j, i, l) = R::rgamma(2, 1);
      }
    }
  }

  // Make Gamma cube
  arma::field<arma::cube> Gamma(10000,3);
  for(int m = 0; m < 10000; m++){
    for(int j = 0; j < 3; j++){
      Gamma(m,j) = arma::ones(80,2,5);
      for(int l = 0; l < 2; l++){
        for(int i = 0; i < 5; i++){
          for(int k = 0; k < 80; k++){
            Gamma(m,j)(k,l,i) =  R::rgamma(1.5, 1/1.5);
          }
        }
      }
    }
  }

  // Make Phi matrix
  arma::field<arma::cube> xi(10000, 3);
  arma::field<arma::cube> delta(10000,1);
  for(int m = 0; m < 10000; m++){
    delta(m,0) = arma::ones(3,5,2);
  }
  double tau = 1;
  for(int m = 0; m < 10000; m++){
    for(int j = 0; j < 3; j++){
      xi(m,j) = arma::zeros(80,2,5);
      for(int l = 0; l < 2; l++){
        tau = 1;
        for(int i = 0; i < 5; i++){
          tau = tau * Delta(j, i, l);
          for(int k = 0; k < 80; k++){
            xi(m,j)(k,l,i) = R::rnorm(0,  std::pow(Gamma(m,j)(k,l,i)*tau, -0.5));
          }
        }
      }
    }
    BayesFMMM::updateDeltaXi(xi, Gamma,  A_xi, m, 10000, delta);
  }

  arma::cube Delta_est = arma::zeros(3, 5, 2);
  arma::cube Delta_trace = arma::zeros(10000, 5, 2);
  arma::vec ph = arma::zeros(10000);
  for(int l = 0; l < 2; l++){
    for(int j = 0; j < 3; j++){
      for(int i = 0; i < 5; i++){
        for(int m = 0; m < 10000; m++){
          ph(m) = delta(m, 0)(j, i, l);
          Delta_trace(m,i,l) = delta(m,0)(0,i,l);
        }
        Delta_est(j, i, l) = arma::median(ph);
      }
    }
  }

  arma::field<arma::cube> mod (2,1);
  mod(0,0) = Delta_est;
  mod(1,0) = Delta;
  return mod;
}

// Tests updating A
//
arma::field<arma::cube> TestUpdateAXi(){
  double a_1 = 2;
  double a_2 = 3;
  arma::cube delta = arma::zeros(3, 5, 2);
  double alpha1 = 2;
  double beta1 = 1;
  double alpha2 = 3;
  double beta2 = 1;
  arma::field<arma::cube> A(1000,1);
  for(int i = 0; i < 1000; i++){
    A(i,0) = arma::ones(3,2,2);
  }

  for(int i = 0; i < 1000; i++){
    for(int l = 0; l < 3; l++){
      for(int j = 0; j < 5; j++){
        for(int d = 0; d < 2; d++){
          if(j == 0){
            delta(l,j,d) = 2;
          }else{
            delta(l,j,d) = 3;
          }
        }
      }
    }
    BayesFMMM::updateAXi(alpha1, beta1, alpha2, beta2, delta, 1, 1, i, 1000, A);
  }
  arma::cube A_est = arma::zeros(3, 2, 2);
  arma::vec ph = arma::zeros(1000);
  for(int j = 0; j < 3; j++){
    for(int i = 0; i < 2; i++){
      for(int d = 0; d < 2; d++){
        for(int m =0; m < 1000; m++){
          ph(m) = A(m,0)(j, i, d);
        }
        A_est(j,i,d) = arma::median(ph);
      }

    }
  }
  arma::cube A_true = arma::zeros(3,2,2);
  for(int i = 0; i < 2; i++){
    A_true.slice(i) = {{2,3}, {2,3}, {2,3}};
  }
  arma::field<arma::cube> mod(2,1);
  mod(0,0) = A_est;
  mod(1,0) = A_true;

  return mod;
}

context("Unit tests for xi parameters") {
  test_that("Sampler for xi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateXiCovariateAdj();
    bool similar = true;

    for(int l = 0; l < 3; l++){
      arma::cube est = x(0,l);
      arma::cube truth = x(1,l);
      for(int i = 0; i < est.n_rows; i++){
        for(int j = 0; j < est.n_cols; j++){
          for(int k = 0; k < est.n_slices; k++){
            if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.2){
              similar = false;
            }
          }
        }
      }
    }

    expect_true(similar == true);
  }

  test_that("Sampler for xi parameters under a tempered likelihood"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateXiTemperedCovariateAdj();
    bool similar = true;

    for(int l = 0; l < 3; l++){
      arma::cube est = x(0,l);
      arma::cube truth = x(1,l);
      for(int i = 0; i < est.n_rows; i++){
        for(int j = 0; j < est.n_cols; j++){
          for(int k = 0; k < est.n_slices; k++){
            if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.2){
              similar = false;
            }
          }
        }
      }
    }

    expect_true(similar == true);
  }

  test_that("Sampler for xi parameters under the multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateXiMVCovariateAdj();
    bool similar = true;

    for(int l = 0; l < 3; l++){
      arma::cube est = x(0,l);
      arma::cube truth = x(1,l);
      for(int i = 0; i < est.n_rows; i++){
        for(int j = 0; j < est.n_cols; j++){
          for(int k = 0; k < est.n_slices; k++){
            if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.2){
              similar = false;
            }
          }
        }
      }
    }

    expect_true(similar == true);
  }

  test_that("Sampler for xi parameters under the multivariate tempered likelihood"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateXiTemperedMVCovariateAdj();
    bool similar = true;

    for(int l = 0; l < 3; l++){
      arma::cube est = x(0,l);
      arma::cube truth = x(1,l);
      for(int i = 0; i < est.n_rows; i++){
        for(int j = 0; j < est.n_cols; j++){
          for(int k = 0; k < est.n_slices; k++){
            if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.2){
              similar = false;
            }
          }
        }
      }
    }

    expect_true(similar == true);
  }

  test_that("Sampler for gamma_xi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateGammaXi();
    bool similar = true;

    for(int l = 0; l < 3; l++){
      arma::cube est = x(0,l);
      arma::cube truth = x(1,l);
      for(int i = 0; i < est.n_rows; i++){
        for(int j = 0; j < est.n_cols; j++){
          for(int k = 0; k < est.n_slices; k++){
            if(std::abs(est(i,j,k) - truth(i,j,k)) > 0.5){
              similar = false;
            }
          }
        }
      }
    }

    expect_true(similar == true);
  }

  test_that("Sampler for Delta_xi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateDeltaXi();
    arma::cube est = x(0,0);
    arma::cube truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        for(int k = 0; k < est.n_slices; k++){
          if(std::abs(est(i,j,k) - truth(i,j,k)) > 2){
            similar = false;
          }
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for A_xi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::cube> x = TestUpdateAXi();
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

}
