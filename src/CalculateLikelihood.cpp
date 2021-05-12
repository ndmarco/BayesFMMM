#include <RcppArmadillo.h>
#include <cmath>

//' Calculates the likelihood of the model
//'
//' @name calcLikelihood
//' @param y_obs Field of vectors containing observed time points
//' @param y_star Field of matrices containing unobserved time points for all iterations
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param B_star Field of matrices containing basis functions evaluated at unobserved time points
//' @param nu Matrix containing current nu parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param iter Int containing current MCMC iteration
//' @param sigma Double containing current sigma parameter
double calcLikelihood(const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& y_star,
                      const arma::field<arma::mat>& B_obs,
                      const arma::field<arma::mat>& B_star,
                      const arma::mat& nu,
                      const arma::cube& Phi,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const int& iter,
                      const double& sigma){
  double log_lik = 0;
  double mean = 0;
  for(int i = 0; i < Z.n_rows; i++){
    for(int l = 0; l < y_obs(i,0).n_elem; l++){
      mean = 0;
      for(int k = 0; k < Z.n_cols; k++){
        if(Z(i,k) != 0){
          mean = mean + arma::dot(nu.row(k), B_obs(i,0).row(l));
          for(int n = 0; n < Phi.n_slices; n++){
            mean = mean + chi(i,n) * arma::dot(Phi.slice(n).row(k),
                              B_obs(i,0).row(l));
          }
        }
      }
      log_lik = log_lik + R::dnorm(y_obs(i,0)(l), mean, sigma, true);
    }
    if(y_star(i,0).n_elem > 0){
      for(int l = 0; l < y_star(i,0).n_cols; l++){
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
        log_lik = log_lik + R::dnorm(y_star(i,0)(iter, l), mean, sigma, true);
      }
    }
  }
  return log_lik;
}

