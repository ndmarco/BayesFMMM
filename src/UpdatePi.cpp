#include <RcppArmadillo.h>
#include <cmath>

//' Updates pi
//'
//' @name UpdatePi
//' @param alpha Double that is the hyperparameter
//' @param iter Int containing current MCMC iteration
//' @param tot_mcmc_iters Int  containing total number of MCMC iterations
//' @param Z Matrix that contains the current values of the binary matrix
//' @param pi Matrix containg all samples of Pi
void updatePi(const double& alpha,
              const arma::mat& Z,
              const int& iter,
              const int& tot_mcmc_iters,
              arma::mat& pi){
  for(int l = 0; l < Z.n_cols; l++){
    pi(l, iter) = R::rbeta((alpha/ Z.n_cols) + arma::accu(Z.col(l)), Z.n_rows -
      arma::accu(Z.col(l)) + 1);
  }
  if(iter < (tot_mcmc_iters -1)){
    pi.col(iter + 1) = pi.col(iter);
  }
}

//' Converts from the transformed space to the original parameter space
//'
//' @name convert_pi_tilde_pi
//' @param Z_tilde Vector containing parameters in the transformed space
//' @param Z Vector containing placeholder for variables in the untransformed space

void convert_pi_tilde_pi(const arma::vec& pi_tilde,
                         arma::vec& pi)
{
  double total_sum = 0;
  for(int i = 0; i < pi.n_elem; i++){
    total_sum = total_sum + std::exp(pi_tilde(i));
  }
  for(int i = 0; i < pi.n_elem; i++){
    pi(i) = std::exp(pi_tilde(i)) / total_sum;
  }
}

//' Calculates the log of B(a) function used in the dirichlet distribution
//'
//' @name calc_lB
//' @param alpha Vector containing input to the function
//' @return log_B Double containing the log of the output of the B function

double calc_lB(const arma::vec& alpha){
  double log_B = 0;

  for(int i=0; i < alpha.n_elem; i++){
    log_B = log_B + std::lgamma(alpha(i));
  }
  log_B = log_B - std::lgamma(arma::accu(alpha));

  return log_B;
}

//' Calculates the log pdf of the posterior distribution of pi
//'
//' @name lpdf_pi_PM
//' @param c vector containing hyperparameters
//' @param alpha_3 Double containing current value of alpha_3
//' @param pi Vector containing pi parameters
//' @param Z Matrix containing current Z parameters
//' @return lpdf Double containing the log pdf

double lpdf_pi_PM(const arma::vec& c,
                  const double& alpha_3,
                  const arma::vec& pi,
                  const arma::mat& Z){
  double lpdf = 0;
  for(int k = 0; k < pi.n_elem; k++){
    lpdf = lpdf + ((c(k) - 1) * std::log(pi(k)));
    for(int i = 0; i <  Z.n_rows; i++){
      lpdf = lpdf + (((alpha_3 * pi(k)) - 1) * std::log(Z(i,k)));
    }
    lpdf = lpdf - (Z.n_rows * calc_lB(alpha_3 * pi));
  }

  return lpdf;
}


//' Updates pi for the partial membership model
//'
//' @name UpdatePi_PM
//' @param alpha_3 Double containing the current value of alpha_3
//' @param Z Matrix containing current value of Z parameters
//' @param c Vector containing hyperparameters
//' @param iter Int containing current MCMC iteration
//' @param tot_mcmc_iters Int  containing total number of MCMC iterations
//' @param pi_tilde Matrix containing all transformed parameters of pi
//' @param pi Matrix containing all values for pi values

void updatePi_PM(const double& alpha_3,
                 const arma::mat& Z,
                 const arma::vec& c,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 const double& sigma_pi,
                 arma::mat& pi_tilde,
                 arma::vec& pi_ph,
                 arma::vec& pi_tilde_ph,
                 arma::mat& pi){

  for(int i=0; i < pi.n_rows; i++){
    pi_tilde_ph(i) = pi_tilde(i, iter) + R::rnorm(0, sigma_pi);
  }
  convert_pi_tilde_pi(pi_tilde_ph, pi_ph);

  // calculate proposal log pdf
  double lpdf_new = lpdf_pi_PM(c, alpha_3, pi_ph, Z);

  // calculate current state log pdf
  double lpdf_old = lpdf_pi_PM(c, alpha_3, pi.col(iter), Z);

  double acceptance_prob = lpdf_new - lpdf_old;
  double rand_unif_var = R::runif(0,1);

  if(std::log(rand_unif_var) < acceptance_prob){
    // Accept new state and update parameters
    pi.col(iter) = pi_ph;
    pi_tilde.col(iter) = pi_tilde_ph;
  }


  if((tot_mcmc_iters - 1) > iter){
    pi.col(iter+1) = pi.col(iter);
    pi_tilde.col(iter+1) = pi_tilde.col(iter);
  }
}


