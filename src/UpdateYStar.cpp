#include <RcppArmadillo.h>
#include <cmath>

//' Calculates the log pdf of the tempered transition MCMC
//'
//' @name lpdfYStar
//' @param beta_i Double containing the current temperature
//' @param y_star Double containing the proposed y_star
//' @param mean Double containing the mean parameter
//' @param sigma Double containing the current sigma value

double lpdfYStar(const double& beta_i,
                 const double& y_star,
                 const double& mean,
                 const double& sigma){
  double lpdf = (- beta_i / 2)* std::log(sigma) - ((beta_i / (2 * sigma)) *
    std::pow(y_star - mean, 2));
  return lpdf;
}


//' Updates the Y_star parameters
//'
//' @name updateYStar
//' @param B_star Field of matrices containing basis functions evaluated at unobserved time points
//' @param nu Matrix containing current nu parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param sigma  Double containing current sigma parameter
//' @param iter Int containing current MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param y_star Field of Matrices containing y_star for all mcmc iterations

void updateYStar(const arma::field<arma::mat>& B_star,
                  const arma::mat& nu,
                  const arma::cube& Phi,
                  const arma::mat& Z,
                  const arma::mat& chi,
                  const double& sigma,
                  const int& iter,
                  const int& tot_mcmc_iters,
                  arma::field<arma::mat>& y_star){
  double mean = 0;
  for(int i = 0; i < Z.n_rows; i++){
    if(B_star(i,0).n_elem > 0){
      for(int l = 0; l < B_star(i,0).n_rows; l++){
        mean = 0;
        for(int k = 0; k < Z.n_cols; k++){
          if(Z(i,k) != 0){
            mean = mean + arma::dot(nu.row(k), B_star(i,0).row(l));
            for(int n = 0; n < Phi.n_slices; n++){
              mean = mean + chi(i,n) * arma::dot(Phi.slice(n).row(k),
                                B_star(i,0).row(l));
            }
          }
        }
        y_star(i,0)(iter, l) = R::rnorm(mean, std::sqrt(sigma));
      }
    }
    if(iter < (tot_mcmc_iters - 1)){
      y_star(i,0).row(iter + 1) = y_star(i,0).row(iter);
    }
  }
}

//' Updates the Y_star parameters using Tempered Transitions
//'
//' @name updateYStarTempered
//' @param beta_i Double containing current temperature
//' @param B_star Field of matrices containing basis functions evaluated at unobserved time points
//' @param nu Matrix containing current nu parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param sigma  Double containing current sigma parameter
//' @param iter Int containing current MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param y_star Field of Matrices containing y_star for all mcmc iterations

void updateYStarTempered(const double& beta_i,
                         const arma::field<arma::mat>& B_star,
                         const arma::mat& nu,
                         const arma::cube& Phi,
                         const arma::mat& Z,
                         const arma::mat& chi,
                         const double& sigma,
                         const int& iter,
                         const int& tot_mcmc_iters,
                         arma::field<arma::mat>& y_star){
  double rand_unif_var = 0;
  double A = 0;
  double lpdf_proposal = 0;
  double lpdf_old = 0;
  double q_old = 0;
  double q_new = 0;
  double y_star_proposal = 0;
  double mean = 0;
  for(int i = 0; i < Z.n_rows; i++){
    if(B_star(i,0).n_elem > 0){
      for(int l = 0; l < B_star(i,0).n_rows; l++){
        mean = 0;
        for(int k = 0; k < Z.n_cols; k++){
          if(Z(i,k) != 0){
            mean = mean + arma::dot(nu.row(k), B_star(i,0).row(l));
            for(int n = 0; n < Phi.n_slices; n++){
              mean = mean + chi(i,n) * arma::dot(Phi.slice(n).row(k),
                                B_star(i,0).row(l));
            }
          }
        }
        y_star_proposal = R::rnorm(mean, std::sqrt(sigma / beta_i));
        lpdf_old = lpdfYStar(beta_i, y_star(i,0)(iter, l), mean, sigma);
        lpdf_proposal = lpdfYStar(beta_i, y_star_proposal, mean, sigma);
        q_old = R::dnorm(y_star(i,0)(iter, l), mean, std::sqrt(sigma / beta_i),
                         true);
        q_new = R::dnorm4(y_star_proposal, mean, std::sqrt(sigma / beta_i),
                          true);
        A = lpdf_proposal + q_old - lpdf_old - q_new;
        rand_unif_var = R::runif(0,1);
        if(std::log(rand_unif_var) < A){
          // Accept new state and update parameters
          y_star(i,0)(iter, l) = y_star_proposal;
        }
      }
    }
    if(iter < (tot_mcmc_iters - 1)){
      y_star(i,0).row(iter + 1) = y_star(i,0).row(iter);
    }
  }
}
