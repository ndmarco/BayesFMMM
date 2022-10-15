#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include <BayesFMMM.h>

// Tests updating chi
//
arma::field<arma::mat> TestUpdateChi(){
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

  arma::field<arma::mat> B_star(40,1);

  for(int i = 0; i < 40; i++){
    B_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++){
    Phi.slice(i) = (3-i) * arma::randn<arma::mat>(3,8);
  }
  double sigma_sq = 0.0001;

  // Make chi matrix
  arma::mat chi(40, 3, arma::fill::randn);


  arma::mat Z(40, 3);
  arma::vec c(3, arma::fill::ones);

  // setting alpha_3 = 10
  arma:: vec alpha = c * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha).t();
  }

  arma::field<arma::vec> y_obs(40, 1);
  arma::vec mean = arma::zeros(8);

  arma::cube chi_samp(40, 3, 500, arma::fill::randn);
  for(int i = 0; i < 500; i++){
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
    BayesFMMM::updateChi(y_obs, B_obs, Phi, nu, Z, sigma_sq, i, 500,
              chi_samp);
  }

  arma::vec chi_ph = arma::zeros(200);
  arma::mat chi_est = arma::zeros(40, 3);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 40; j++){
      for(int l = 300; l < 500; l++){
        chi_ph(l - 300) = chi_samp(j, i, l);
      }
      chi_est(j,i) = arma::median(chi_ph);
    }
  }

  arma::field<arma::mat> mod(2,1);
  mod(0,0) = chi_est;
  mod(1,0) = chi;

  return mod;
}

// Tests updating chi using tempered transitions
//
arma::field<arma::mat> TestUpdateChiTempered(){
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

  arma::field<arma::mat> B_star(40,1);

  for(int i = 0; i < 40; i++){
    B_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
  }

  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++){
    Phi.slice(i) = (3-i) * arma::randn<arma::mat>(3,8);
  }
  double sigma_sq = 0.0001;

  // Make chi matrix
  arma::mat chi(40, 3, arma::fill::randn);


  arma::mat Z(40, 3);
  arma::vec c(3, arma::fill::ones);

  // setting alpha_3 = 10
  arma:: vec alpha = c * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha).t();
  }

  arma::field<arma::vec> y_obs(40, 1);
  arma::vec mean = arma::zeros(8);

  arma::cube chi_samp(40, 3, 500, arma::fill::randn);
  for(int i = 0; i < 500; i++){
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
    BayesFMMM::updateChiTempered(0.5, y_obs, B_obs, Phi, nu, Z, sigma_sq, i, 500,
                         chi_samp);
  }

  arma::vec chi_ph = arma::zeros(200);
  arma::mat chi_est = arma::zeros(40, 3);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 40; j++){
      for(int l = 300; l < 500; l++){
        chi_ph(l - 300) = chi_samp(j, i, l);
      }
      chi_est(j,i) = arma::median(chi_ph);
    }
  }

  arma::field<arma::mat> mod(2,1);
  mod(0,0) = chi_est;
  mod(1,0) = chi;

  return mod;
}

// Tests updating Nu for multivariate model
//
// @name TestUpdateNu
arma::field<arma::mat> TestUpdateChiMV(){
  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++){
    Phi.slice(i) = (3-i) * arma::randn<arma::mat>(3,8);
  }
  double sigma_sq = 0.0001;

  // Make chi matrix
  arma::mat chi(40, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(40, 3);
  arma::mat alpha(40,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::mat y_obs = arma::zeros(40, 8);
  arma::vec mean = arma::zeros(8);

  arma::cube chi_samp(40, 3, 500, arma::fill::randn);
  for(int i = 0; i < 500; i++){
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
    BayesFMMM::updateChiMV(y_obs, Phi, nu, Z, sigma_sq, i, 500,
                         chi_samp);
  }

  arma::vec chi_ph = arma::zeros(200);
  arma::mat chi_est = arma::zeros(40, 3);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 40; j++){
      for(int l = 300; l < 500; l++){
        chi_ph(l - 300) = chi_samp(j, i, l);
      }
      chi_est(j,i) = arma::median(chi_ph);
    }
  }

  arma::field<arma::mat> mod(2,1);
  mod(0,0) = chi_est;
  mod(1,0) = chi;

  return mod;
}

// Tests updating Nu for using tempered transitions for a multivariate model
//
// @name TestUpdateNu
arma::field<arma::mat> TestUpdateChiTemperedMV(){
  // Make nu matrix
  arma::mat nu(3,8);
  nu = {{2, 0, 1, 0, 0, 0, 1, 3},
  {1, 3, 0, 2, 0, 0, 3, 0},
  {5, 2, 5, 0, 3, 4, 1, 0}};


  // Make Phi matrix
  arma::cube Phi(3,8,3);
  for(int i=0; i < 3; i++){
    Phi.slice(i) = (3-i) * arma::randn<arma::mat>(3,8);
  }
  double sigma_sq = 0.0001;

  // Make chi matrix
  arma::mat chi(40, 3, arma::fill::randn);


  // Make Z matrix
  arma::mat Z(40, 3);
  arma::mat alpha(40,3, arma::fill::ones);
  alpha = alpha * 10;
  for(int i = 0; i < Z.n_rows; i++){
    Z.row(i) = BayesFMMM::rdirichlet(alpha.row(i).t()).t();
  }

  arma::mat y_obs = arma::zeros(40, 8);
  arma::vec mean = arma::zeros(8);

  arma::cube chi_samp(40, 3, 500, arma::fill::randn);
  for(int i = 0; i < 500; i++){
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
    BayesFMMM::updateChiTemperedMV(0.5, y_obs, Phi, nu, Z, sigma_sq, i, 500,
                           chi_samp);
  }

  arma::vec chi_ph = arma::zeros(200);
  arma::mat chi_est = arma::zeros(40, 3);
  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 40; j++){
      for(int l = 300; l < 500; l++){
        chi_ph(l - 300) = chi_samp(j, i, l);
      }
      chi_est(j,i) = arma::median(chi_ph);
    }
  }

  arma::field<arma::mat> mod(2,1);
  mod(0,0) = chi_est;
  mod(1,0) = chi;

  return mod;
}

// Tests updating chi
//
arma::field<arma::mat> TestUpdateChiCovariateAdj(){
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
  arma::field<arma::cube> xi(500,3);
  for(int i = 0; i < 3; i++){
    xi(0,i) = arma::zeros(8,2,2);
    for(int j =0; j < 2; j++){
      xi(0,i).slice(j) = (2-i) * arma::randn<arma::mat>(8,2);
    }
  }
  for(int i = 1; i < 500; i++){
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
  arma::cube chi_samp(40, 2, 500, arma::fill::randn);
  for(int i = 0; i < 500; i++){
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
    BayesFMMM::updateChiCovariateAdj(y_obs, B_obs, Phi, xi, nu, eta, Z,
                                     sigma_sq, i, 500, X, chi_samp);
  }

  arma::vec chi_ph = arma::zeros(200);
  arma::mat chi_est = arma::zeros(40, 2);
  for(int i = 0; i < 2; i++){
    for(int j = 0; j < 40; j++){
      for(int l = 300; l < 500; l++){
        chi_ph(l - 300) = chi_samp(j, i, l);
      }
      chi_est(j,i) = arma::median(chi_ph);
    }
  }

  arma::field<arma::mat> mod(2,1);
  mod(0,0) = chi_est;
  mod(1,0) = chi;

  return mod;
}

// Tests updating chi for covariate adjusted model
//
arma::field<arma::mat> TestUpdateChiTemperedCovariateAdj(){
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
  arma::field<arma::cube> xi(500,3);
  for(int i = 0; i < 3; i++){
    xi(0,i) = arma::zeros(8,2,2);
    for(int j =0; j < 2; j++){
      xi(0,i).slice(j) = (2-i) * arma::randn<arma::mat>(8,2);
    }
  }
  for(int i = 1; i < 500; i++){
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
  arma::cube chi_samp(40, 2, 500, arma::fill::randn);
  for(int i = 0; i < 500; i++){
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
    BayesFMMM::updateChiTemperedCovariateAdj(0.5, y_obs, B_obs, Phi, xi, nu, eta,
                                             Z, sigma_sq, i, 500, X, chi_samp);
  }

  arma::vec chi_ph = arma::zeros(200);
  arma::mat chi_est = arma::zeros(40, 2);
  for(int i = 0; i < 2; i++){
    for(int j = 0; j < 40; j++){
      for(int l = 300; l < 500; l++){
        chi_ph(l - 300) = chi_samp(j, i, l);
      }
      chi_est(j,i) = arma::median(chi_ph);
    }
  }

  arma::field<arma::mat> mod(2,1);
  mod(0,0) = chi_est;
  mod(1,0) = chi;

  return mod;
}

// Tests updating Nu for the covariate adjusted multivariate model
//
// @name TestUpdateNu
arma::field<arma::mat> TestUpdateChiMVCovariateAdj(){
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
  arma::field<arma::cube> xi(500,3);
  for(int i = 0; i < 3; i++){
    xi(0,i) = arma::zeros(8,2,2);
    for(int j =0; j < 2; j++){
      xi(0,i).slice(j) = (2-i) * arma::randn<arma::mat>(8,2);
    }
  }
  for(int i = 1; i < 500; i++){
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
  arma::cube chi_samp(40, 2, 500, arma::fill::randn);

  for(int i = 0; i < 500; i++){
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
    BayesFMMM::updateChiMVCovariateAdj(y_obs, Phi, xi, nu, eta, Z, sigma_sq, i,
                                       500, X, chi_samp);
  }

  arma::vec chi_ph = arma::zeros(200);
  arma::mat chi_est = arma::zeros(40, 2);
  for(int i = 0; i < 2; i++){
    for(int j = 0; j < 40; j++){
      for(int l = 300; l < 500; l++){
        chi_ph(l - 300) = chi_samp(j, i, l);
      }
      chi_est(j,i) = arma::median(chi_ph);
    }
  }

  arma::field<arma::mat> mod(2,1);
  mod(0,0) = chi_est;
  mod(1,0) = chi;

  return mod;
}

// Tests updating Nu for the covariate adjusted multivariate model
//
// @name TestUpdateNu
arma::field<arma::mat> TestUpdateChiTemperedMVCovariateAdj(){
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
  arma::field<arma::cube> xi(500,3);
  for(int i = 0; i < 3; i++){
    xi(0,i) = arma::zeros(8,2,2);
    for(int j =0; j < 2; j++){
      xi(0,i).slice(j) = (2-i) * arma::randn<arma::mat>(8,2);
    }
  }
  for(int i = 1; i < 500; i++){
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
  arma::cube chi_samp(40, 2, 500, arma::fill::randn);

  for(int i = 0; i < 500; i++){
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
    BayesFMMM::updateChiTemperedMVCovariateAdj(0.5, y_obs, Phi, xi, nu, eta, Z,
                                               sigma_sq, i, 500, X, chi_samp);
  }

  arma::vec chi_ph = arma::zeros(200);
  arma::mat chi_est = arma::zeros(40, 2);
  for(int i = 0; i < 2; i++){
    for(int j = 0; j < 40; j++){
      for(int l = 300; l < 500; l++){
        chi_ph(l - 300) = chi_samp(j, i, l);
      }
      chi_est(j,i) = arma::median(chi_ph);
    }
  }

  arma::field<arma::mat> mod(2,1);
  mod(0,0) = chi_est;
  mod(1,0) = chi;

  return mod;
}

context("Unit tests for Chi parameters") {
  test_that("Sampler for Chi parameters"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateChi();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Chi parameters using tempered transitions"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateChiTempered();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Chi parameters for multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateChiMV();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Chi parameters using tempered transitions for multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateChiTemperedMV();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Chi parameters under the covariate adjusted model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateChiCovariateAdj();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }
  test_that("Sampler for Chi parameters under the covariate adjusted model using a tempered likelihood"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateChiTemperedCovariateAdj();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Chi parameters for the covariate adjusted multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateChiMVCovariateAdj();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

  test_that("Sampler for Chi parameters for the tempered likelihood covariate adjusted multivariate model"){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(1);
    arma::field<arma::mat> x = TestUpdateChiTemperedMVCovariateAdj();
    arma::mat est = x(0,0);
    arma::mat truth = x(1,0);
    bool similar = true;
    for(int i = 0; i < est.n_rows; i++){
      for(int j = 0; j < est.n_cols; j++){
        if(std::abs(est(i,j) - truth(i,j)) > 0.2){
          similar = false;
        }
      }
    }
    expect_true(similar == true);
  }

}
