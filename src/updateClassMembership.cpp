#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]


//' Gets log-pdf of z_i given zeta_{-z_i}
//'
//' @name lpdf_z
//' @param y_obs Vector containing y at observed time points
//' @param y_star Vector containing y at unobserved time points
//' @param B_obs Matrix containing basis functions evaluated at observed time points
//' @param B_star Matrix containing basis functions evaluated at unobserved time points
//' @param Phi Cube containing Phi parameters
//' @param nu Matrix containing nu parameters
//' @param pi vector containing the elements of pi
//' @param Z Vector containing the ith row of Z
//' @param sigma_sq double containing the sigma_sq variable
//' @return lpdf_z double contianing the log-pdf

double lpdf_z(const arma::vec& y_obs,
              const arma::mat& y_star,
              const arma::mat& B_obs,
              const arma::mat& B_star,
              const arma::cube& Phi,
              const arma::mat& nu,
              const arma::rowvec& chi,
              const arma::vec& pi,
              const arma::rowvec& Z,
              const int& num,
              const double& sigma_sq){
  double lpdf = 0;
  double mean = 0;

  for(int l = 0; l < pi.n_elem; l++){
    lpdf = lpdf + Z(l) * log(pi(l)) + (1 - Z(l)) *  log(1 - pi(l));
  }

  for(int l = 0; l < B_obs.n_rows; l++){
    mean = 0;
    for(int k = 0; k < pi.n_elem; k++){
      mean = mean + Z(k) * arma::dot(nu.row(k), B_obs.row(l).t());
      for(int n = 0; n < Phi.n_slices; n++){
         mean = mean + Z(k) * chi(n) * arma::dot(Phi.slice(n).row(k),
                         B_obs.row(l).t());
      }
    }
    lpdf = lpdf - (std::pow(y_obs(l) - mean, 2.0) / (2 * sigma_sq));
  }

  // Check to see if there are unobserved time points of interest
  if(B_star.n_rows > 0){
    for(int l = 0; l < B_star.n_rows; l++){
      mean = 0;
      for(int k = 0; k < pi.n_elem; k++){
        mean = mean + Z(k) * arma::dot(nu.row(k), B_star.row(l).t());
        for(int n = 0; n < Phi.n_slices; n++){
          mean = mean + Z(k) * chi(n) * arma::dot(Phi.slice(n).row(k),
                          B_star.row(l).t());
        }
      }
      lpdf = lpdf - (std::pow(y_star(num,l) - mean, 2.0) / (2 * sigma_sq));
    }
  }

  return lpdf;
}


//' Updates the Z Matrix
//'
//' @name UpdateZ
//' @param y_obs Field of Vectors containing y at observed time points
//' @param y_star Field of Matrices containing y at unobserved time points at all mcmc iterations
//' @param B_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param B_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Phi Cube containing Phi parameters
//' @param nu Matrix containing nu parameters
//' @param pi Vector containing the elements of pi
//' @param sigma_sq Double containing the sigma_sq variable
//' @param rho Double containing hyperparameter for proposal of new z_i state
//' @param iter Int containing current mcmc iteration
//' @param tot_mcmc_iters Int containing total number of mcmc iterations
//' @param Z_ph Matrix that acts as a placeholder for Z
//' @param Z Cube that contains all past, current, and future MCMC draws

void updateZ(const arma::field<arma::vec>& y_obs,
             const arma::field<arma::mat>& y_star,
             const arma::field<arma::mat>& B_obs,
             const arma::field<arma::mat>& B_star,
             const arma::cube& Phi,
             const arma::mat& nu,
             const arma::mat& chi,
             const arma::vec& pi,
             const double& sigma_sq,
             const double& rho,
             const int& iter,
             const int& tot_mcmc_iters,
             arma::mat& Z_ph,
             arma::cube& Z){
  double z_lpdf = 0;
  double z_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;

  Z_ph = Z.slice(iter);
  for(int i = 0; i < Z.n_rows; i++){
    for(int l = 0; l < Z.n_cols; l++){
      // Propose new state
      Z_ph(i,l) = R::rbinom(1, Z.slice(iter)(i,l) * rho +
        ((1 - Z.slice(iter)(i,l)) * (1 -rho)));
    }
    // Get old state log pdf
    z_lpdf = lpdf_z(y_obs(i,0), y_star(i,0), B_obs(i,0), B_star(i,0),
                    Phi, nu, chi.row(i), pi, Z.slice(iter).row(i), i, sigma_sq);

    // Get new state log pdf
    z_new_lpdf = lpdf_z(y_obs(i,0), y_star(i,0), B_obs(i,0),
                        B_star(i,0), Phi,  nu, chi.row(i), pi, Z_ph.row(i), i,
                        sigma_sq);
    acceptance_prob = z_new_lpdf - z_lpdf;
    rand_unif_var = R::runif(0,1);

    if(log(rand_unif_var) < acceptance_prob){
      // Accept new state and update parameters
      Z.slice(iter).row(i) = Z_ph.row(i);
    } else{
      Z_ph.row(i) = Z.slice(iter).row(i);
    }
  }
  // Update next iteration
  if(iter < (tot_mcmc_iters - 1)){
    Z.slice(iter + 1) = Z.slice(iter);
  }
}
