#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include <BayesFPMM.h>

//' Function for finding a good initial starting point for nu parameters and Z parameters for functional data, with option for tempered transitions
//'
//' @name BFPMM_Nu_Z_multiple_try
//' @param tot_mcmc_iters Int containing the number of MCMC iterations per try
//' @param n_try Int containing how many different chains are tried
//' @param k Int containing the number of clusters
//' @param Y Field of vectors containing the observed values
//' @param time Field of vectors containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param basis_degree Int containing the degree of B-splines used
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @returns BestChain List containing a summary of the best performing chain
//' @export
// [[Rcpp::export]]
Rcpp::List BFPMM_Nu_Z_multiple_try(const int tot_mcmc_iters,
                                   const int n_try,
                                   const int k,
                                   const arma::field<arma::vec> Y,
                                   const arma::field<arma::vec> time,
                                   const int n_funct,
                                   const int basis_degree,
                                   const int n_eigen,
                                   const arma::vec boundary_knots,
                                   const arma::vec internal_knots){
  splines2::BSpline bspline;
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  for(int i = 0; i < n_funct; i++)
  {
    // Create Bspline object
    bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat{bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  // placeholder
  arma::vec c = arma::ones(k);

  // start MCMC sampling
  Rcpp::List mod1 = BayesFPMM::BFPMM_Nu_Z(Y, time, n_funct, k, basis_degree, n_eigen,
                                          boundary_knots, internal_knots,
                                          tot_mcmc_iters,c, 800, 3, 2, 3, 1, 1, 1000, 1000,
                                          0.05, sqrt(1), sqrt(1), 1, 10, 1, 1);
  arma::vec ph = mod1["loglik"];
  double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

  for(int i = 0; i < n_try; i++){
    Rcpp::List modi = BayesFPMM::BFPMM_Nu_Z(Y, time, n_funct, k, basis_degree, n_eigen,
                                            boundary_knots, internal_knots,
                                            tot_mcmc_iters, c, 800, 3, 2, 3, 1, 1, 1000,
                                            1000, 0.05, sqrt(1), sqrt(1), 1,10, 1, 1);
    arma::vec ph1 = modi["loglik"];
    if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
      mod1 = modi;
      min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
    }

  }

  Rcpp::List BestChain =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                             Rcpp::Named("nu", mod1["nu"]),
                                             Rcpp::Named("pi", mod1["pi"]),
                                             Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                             Rcpp::Named("A", mod1["A"]),
                                             Rcpp::Named("delta", mod1["delta"]),
                                             Rcpp::Named("sigma", mod1["sigma"]),
                                             Rcpp::Named("tau", mod1["tau"]),
                                             Rcpp::Named("Z", mod1["Z"]),
                                             Rcpp::Named("loglik", mod1["loglik"]));

  return BestChain;
}

//' Estimates the initial starting point of the rest of the parameters given an initial starting point for Z and nu for functional data
//'
//' @name BFPMM_Theta_Est
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param Z_samp Cube containing initial chain of Z parameters from BFPMM_Nu_Z_multiple_try
//' @param nu_samp Cube containing initial chain of nu parameters from BFPMM_Nu_Z_multiple_try
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param k Int containing the number of clusters
//' @param Y Field of vectors containing the observed values
//' @param time Field of vectors containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param basis_degree Int containing the degree of B-splines used
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @returns BestChain List containing a summary of the chain conditioned on nu and Z
//' @export
// [[Rcpp::export]]
Rcpp::List BFPMM_Theta_Est(const int tot_mcmc_iters,
                           const arma::cube Z_samp,
                           const arma::cube nu_samp,
                           double burnin_prop,
                           const int k,
                           const arma::field<arma::vec> Y,
                           const arma::field<arma::vec> time,
                           const int n_funct,
                           const int basis_degree,
                           const int n_eigen,
                           const arma::vec boundary_knots,
                           const arma::vec internal_knots){

  splines2::BSpline bspline;
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  for(int i = 0; i < n_funct; i++)
  {
    // Create Bspline object
    bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat{bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  arma::vec c = arma::ones(k);

  int n_nu = nu_samp.n_slices;
  arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
  arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
  arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  for(int i = 0; i < Z_est.n_cols; i++){
    for(int j = 0; j < Z_est.n_rows; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
      }
      Z_est(j,i) = arma::median(ph_Z);
    }
    for(int j = 0; j < nu_samp.n_cols; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
      }
      nu_est(i,j) = arma::median(ph_nu);
    }
  }

  // normalize
  for(int i = 0; i < Z_est.n_rows; i++){
    for(int j = 0; j < Z_est.n_cols; j++){
      Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
    }
  }

  // start MCMC sampling
  Rcpp::List mod1 = BayesFPMM::BFPMM_Theta(Y, time, n_funct, k, basis_degree, n_eigen,
                                           boundary_knots, internal_knots, tot_mcmc_iters,
                                           c, 1, 3, 2, 3, 1, 1, 1000, 1000, 0.05,
                                           sqrt(1), sqrt(1), 1, 5, 1, 1, Z_est, nu_est);

  Rcpp::List BestChain =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                             Rcpp::Named("chi", mod1["chi"]),
                                             Rcpp::Named("A", mod1["A"]),
                                             Rcpp::Named("delta", mod1["delta"]),
                                             Rcpp::Named("sigma", mod1["sigma"]),
                                             Rcpp::Named("tau", mod1["tau"]),
                                             Rcpp::Named("gamma", mod1["gamma"]),
                                             Rcpp::Named("Phi", mod1["Phi"]),
                                             Rcpp::Named("Nu_est", nu_est),
                                             Rcpp::Named("loglik", mod1["loglik"]));

  return BestChain;
}

//' Performs MCMC for functional data, with optional tempered transitions, using user specified starting points
//'
//' @name BFPMM_warm_start
//' @param beta_N_t Double containing the maximum weight for tempered transisitons
//' @param N_t Int containing total number of tempered transitions. If no tempered transitions are desired, pick a small integer
//' @param n_temp_trans Int containing how often tempered transitions are performed. If no tempered transitions are desired, pick a integer larger than tot_mcmc_iters
//' @param tot_mcmc_iters Int containing the number of MCMC iterations
//' @param r_stored_iters Int containing number of MCMC iterations stored in memory before writing to directory
//' @param Z_samp Cube containing initial chain of Z parameters
//' @param pi_samp Matrix containing initial chain of pi parameters
//' @param alpha_3_samp Vector containing initial chain of alpha_3 parameters
//' @param delta_samp Matrix containing initial chain of delta parameters
//' @param gamma_samp Field of cubes containing initial chain of gamma parameters
//' @param Phi_samp Field of cubes containing initial chain of phi parameters
//' @param A_samp Matrix containing initial chain of A parameters
//' @param nu_samp Cube containing initial chain of nu parameters
//' @param tau_samp Matrix containing initial chain of tau parameters
//' @param sigma_samp Vector containing initial chain of sigma parameters
//' @param chi_samp Cube containing initial chain of chi parameters
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param k Int containing the number of clusters
//' @param Y Field of vectors containing the observed values
//' @param time Field of vectors containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param basis_degree Int containing the degree of B-splines used
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param thinning_num Int containing how often we should save MCMC iterations. Should be a divisible by r_stored_iters and tot_mcmc_iters
//' @param dir String containing directory where the MCMC files should be saved
//' @export
// [[Rcpp::export]]
Rcpp::List BFPMM_warm_start(const double beta_N_t,
                            const int N_t,
                            const int n_temp_trans,
                            const int tot_mcmc_iters,
                            const int r_stored_iters,
                            const arma::cube Z_samp,
                            const arma::mat pi_samp,
                            const arma::vec alpha_3_samp,
                            const arma::mat delta_samp,
                            const arma::field<arma::cube> gamma_samp,
                            const arma::field<arma::cube> Phi_samp,
                            const arma::mat A_samp,
                            const arma::cube nu_samp,
                            const arma::mat tau_samp,
                            const arma::vec sigma_samp,
                            const arma::cube chi_samp,
                            const double burnin_prop,
                            const int k,
                            const arma::field<arma::vec> Y,
                            const arma::field<arma::vec> time,
                            const int n_funct,
                            const int basis_degree,
                            const int n_eigen,
                            const arma::vec boundary_knots,
                            const arma::vec internal_knots,
                            const double thinning_num,
                            const std::string dir){
  splines2::BSpline bspline;
  // Make B_obs
  arma::field<arma::mat> B_obs(n_funct,1);
  for(int i = 0; i < n_funct; i++)
  {
    // Create Bspline object
    bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat{bspline.basis(true)};
    B_obs(i,0) = bspline_mat;
  }

  arma::vec c = arma::ones(k);

  int n_nu = alpha_3_samp.n_elem;

  double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
  arma::vec pi_est = arma::zeros(pi_samp.n_rows);
  arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
  arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
  arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  for(int i = 0; i < Z_est.n_cols; i++){
    pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    for(int j = 0; j < Z_est.n_rows; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
      }
      Z_est(j,i) = arma::median(ph_Z);
    }
    for(int j = 0; j < nu_samp.n_cols; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
      }
      nu_est(i,j) = arma::median(ph_nu);
    }
  }

  // normalize
  for(int i = 0; i < Z_est.n_rows; i++){
    Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
  }

  pi_est = pi_est / arma::accu(pi_est);

  int n_Phi = sigma_samp.n_elem;

  double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
  arma::vec delta_est = arma::zeros(delta_samp.n_rows);
  arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  for(int i = 0; i < delta_samp.n_rows; i++){
    for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
      ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,l);
    }
    delta_est(i) = arma::median(ph_delta);
  }
  arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
  arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
  arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  for(int i = 0; i < Phi_est.n_rows; i++){
    for(int j = 0; j < Phi_est.n_cols; j++){
      for(int m = 0; m < Phi_est.n_slices; m++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);

          ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
        }
        Phi_est(i,j,m) = arma::median(ph_phi);
        gamma_est(i,j,m) = arma::median(ph_gamma);
      }
    }
  }

  arma::vec A_est = arma::zeros(A_samp.n_cols);
  arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  for(int i = 0; i < A_est.n_elem; i++){
    for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
      ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(l,i);
    }
    A_est(i) = arma::median(ph_A);
  }
  arma::vec tau_est = arma::zeros(tau_samp.n_cols);
  arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  for(int i = 0; i < tau_est.n_elem; i++){
    for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
      ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
    }
    tau_est(i) = arma::median(ph_tau);
  }
  arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
  arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
  for(int i = 0; i < chi_est.n_rows; i++){
    for(int j = 0; j < chi_est.n_cols; j++){
      for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
        ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
      }
      chi_est(i,j) = arma::median(ph_chi);
    }
  }

  // start MCMC sampling
  Rcpp::List mod1 = BayesFPMM::BFPMM_MTT_warm_start(Y, time, n_funct, thinning_num, k,
                                                    basis_degree, n_eigen, boundary_knots,
                                                    internal_knots, tot_mcmc_iters,
                                                    r_stored_iters, n_temp_trans,
                                                    c, 800, 3, 2, 3, 1, 1, 1000, 1000, 0.05,
                                                    sqrt(1), sqrt(1), 1, 10, 1, 1, dir,
                                                    beta_N_t, N_t, Z_est, pi_est, alpha_3_est,
                                                    delta_est, gamma_est, Phi_est, A_est,
                                                    nu_est, tau_est, sigma_est, chi_est);

  Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                        Rcpp::Named("nu", mod1["nu"]),
                                        Rcpp::Named("chi", mod1["chi"]),
                                        Rcpp::Named("pi", mod1["pi"]),
                                        Rcpp::Named("alpha_3", mod1["alpha_3"]),
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


//' Reads in armadillo vector and returns it in R format
//'
//' @name ReadVec
//' @param file String containing location where arma vector is stored
//' @export
// [[Rcpp::export]]
arma::vec ReadVec(std::string file){
  arma::vec B;
  B.load(file);
  return B;
}

//' Reads in armadillo matrix and returns it in R format
//'
//' @name ReadMat
//' @param file String containing location where arma matrix is stored
//' @export
// [[Rcpp::export]]
arma::mat ReadMat(std::string file){
  arma::mat B;
  B.load(file);
  return B;
}

//' Reads in armadillo cube and returns it in R format
//'
//' @name ReadCube
//' @param file String containing location where arma cube is stored
//' @export
// [[Rcpp::export]]
arma::cube ReadCube(std::string file){
  arma::cube B;
  B.load(file);
  return B;
}

//' Reads in armadillo field of cubes and returns it in R format
//'
//' @name ReadFieldCube
//' @param file String containing location where arma field is stored
//' @export
// [[Rcpp::export]]
arma::field<arma::cube> ReadFieldCube(std::string file){
  arma::field<arma::cube> B;
  B.load(file);
  return B;
}

//' Reads in armadillo field of matrices and returns it in R format
//'
//' @name ReadFieldMat
//' @param file String containing location where arma field is stored
//' @export
// [[Rcpp::export]]
arma::field<arma::mat> ReadFieldMat(std::string file){
  arma::field<arma::mat> B;
  B.load(file);
  return B;
}

//' Reads in armadillo field of vectors and returns it in R format
//'
//' @name ReadFieldVec
//' @param file String containing location where arma field is stored
//' @export
// [[Rcpp::export]]
arma::field<arma::vec> ReadFieldVec(std::string file){
  arma::field<arma::vec> B;
  B.load(file);
  return B;
}

//' Function for finding a good initial starting point for nu parameters and Z parameters for multivariate functional data, with option for temperered transitions
//'
//' @name BMFPMM_Nu_Z_multiple_try
//' @param tot_mcmc_iters Int containing the number of MCMC iterations per try
//' @param n_try Int containing how many different chains are tried
//' @param k Int containing the number of clusters
//' @param Y Field of vectors containing the observed values
//' @param time field of matrices that contain the observed time points (each column is a dimension)
//' @param n_funct Int containing the number of functions
//' @param basis_degree vector containing the desired basis degree for each dimension
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots matrix containing the boundary knots for each dimension (each row is a dimension)
//' @param internal_knots field of vectors containing the internal knots for each dimension
//' @returns BestChain List containing a summary of the best performing chain
//' @export
// [[Rcpp::export]]
Rcpp::List BMFPMM_Nu_Z_multiple_try(const int tot_mcmc_iters,
                                    const int n_try,
                                    const int k,
                                    const arma::field<arma::vec> Y,
                                    const arma::field<arma::mat> time,
                                    const int n_funct,
                                    const arma::vec basis_degree,
                                    const int n_eigen,
                                    const arma::mat boundary_knots,
                                    const arma::field<arma::vec> internal_knots){

  arma::field<arma::mat> B_obs = BayesFPMM::TensorBSpline(time, n_funct, basis_degree,
                                                          boundary_knots, internal_knots);


  // placeholder
  arma::vec c = arma::ones(k);

  // start MCMC sampling
  Rcpp::List mod1 = BayesFPMM::BMFPMM_Nu_Z(Y, time, n_funct, k, basis_degree, n_eigen,
                                           boundary_knots, internal_knots,
                                           tot_mcmc_iters,c, 800, 3, 2, 3, 1, 1, 1000, 1000,
                                           0.05, sqrt(1), sqrt(1), 1, 10, 1, 1);
  arma::vec ph = mod1["loglik"];
  double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

  for(int i = 0; i < n_try; i++){
    Rcpp::List modi = BayesFPMM::BMFPMM_Nu_Z(Y, time, n_funct, k, basis_degree, n_eigen,
                                             boundary_knots, internal_knots,
                                             tot_mcmc_iters, c, 800, 3, 2, 3, 1, 1, 1000,
                                             1000, 0.05, sqrt(1), sqrt(1), 1,10, 1, 1);
    arma::vec ph1 = modi["loglik"];
    if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
      mod1 = modi;
      min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
    }

  }

  Rcpp::List BestChain =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                             Rcpp::Named("nu", mod1["nu"]),
                                             Rcpp::Named("pi", mod1["pi"]),
                                             Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                             Rcpp::Named("A", mod1["A"]),
                                             Rcpp::Named("delta", mod1["delta"]),
                                             Rcpp::Named("sigma", mod1["sigma"]),
                                             Rcpp::Named("tau", mod1["tau"]),
                                             Rcpp::Named("Z", mod1["Z"]),
                                             Rcpp::Named("loglik", mod1["loglik"]));

  return BestChain;
}

//' Estimates the initial starting point of the rest of the parameters given an initial starting point for Z and nu for multivariate functional data
//'
//' @name BMFPMM_Theta_Est
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param Z_samp Cube containing initial chain of Z parameters from BFPMM_Nu_Z_multiple_try
//' @param nu_samp Cube containing initial chain of nu parameters from BFPMM_Nu_Z_multiple_try
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param k Int containing the number of clusters
//' @param Y Field of vectors containing the observed values
//' @param time Field of vectors containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param basis_degree Int containing the degree of B-splines used
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @returns BestChain List containing a summary of the chain conditioned on nu and Z
//' @export
// [[Rcpp::export]]
Rcpp::List BMFPMM_Theta_Est(const int tot_mcmc_iters,
                           const arma::cube Z_samp,
                           const arma::cube nu_samp,
                           double burnin_prop,
                           const int k,
                           const arma::field<arma::vec> Y,
                           const arma::field<arma::mat> time,
                           const int n_funct,
                           const arma::vec basis_degree,
                           const int n_eigen,
                           const arma::mat boundary_knots,
                           const arma::field<arma::vec> internal_knots){

  arma::field<arma::mat> B_obs = BayesFPMM::TensorBSpline(time, n_funct, basis_degree,
                                                          boundary_knots, internal_knots);

  arma::vec c = arma::ones(k);

  int n_nu = nu_samp.n_slices;
  arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
  arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
  arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
  for(int i = 0; i < Z_est.n_cols; i++){
    for(int j = 0; j < Z_est.n_rows; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
      }
      Z_est(j,i) = arma::median(ph_Z);
    }
    for(int j = 0; j < nu_samp.n_cols; j++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
      }
      nu_est(i,j) = arma::median(ph_nu);
    }
  }

  // start MCMC sampling
  Rcpp::List mod1 = BayesFPMM::BMFPMM_Theta(Y, time, n_funct, k, basis_degree, n_eigen,
                                            boundary_knots, internal_knots, tot_mcmc_iters,
                                            c, 1, 3, 2, 3, 1, 1, 1000, 1000, 0.05,
                                            sqrt(1), sqrt(1), 1, 5, 1, 1, Z_est, nu_est);

  Rcpp::List BestChain =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                             Rcpp::Named("chi", mod1["chi"]),
                                             Rcpp::Named("A", mod1["A"]),
                                             Rcpp::Named("delta", mod1["delta"]),
                                             Rcpp::Named("sigma", mod1["sigma"]),
                                             Rcpp::Named("tau", mod1["tau"]),
                                             Rcpp::Named("gamma", mod1["gamma"]),
                                             Rcpp::Named("Phi", mod1["Phi"]),
                                             Rcpp::Named("Nu_est", nu_est),
                                             Rcpp::Named("loglik", mod1["loglik"]));

  return BestChain;
}


