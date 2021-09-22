#include <RcppArmadillo.h>
#include <cmath>

//' Calculates the log likelihood of the model
//'
//' @name calcLikelihood
//' @param y_obs Field of vectors containing observed time points
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param nu Matrix containing current nu parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param sigma Double containing current sigma parameter
//' @return log_lik Double containing the log likelihood of the model
double calcLikelihood(const arma::field<arma::vec>& y_obs,
                      const arma::field<arma::mat>& B_obs,
                      const arma::mat& nu,
                      const arma::cube& Phi,
                      const arma::mat& Z,
                      const arma::mat& chi,
                      const double& sigma){
  double log_lik = 0;
  double mean = 0;
  for(int i = 0; i < Z.n_rows; i++){
    for(int l = 0; l < y_obs(i,0).n_elem; l++){
      mean = 0;
      for(int k = 0; k < Z.n_cols; k++){
        if(Z(i,k) != 0){
          mean = mean + Z(i,k) * arma::dot(nu.row(k), B_obs(i,0).row(l));
          for(int n = 0; n < Phi.n_slices; n++){
            mean = mean + Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k),
                              B_obs(i,0).row(l));
          }
        }
      }
      log_lik = log_lik + R::dnorm(y_obs(i,0)(l), mean, std::sqrt(sigma), true);
    }
  }
  return log_lik;
}

//' Calculates the likelihood of observing one observation
//'
//' @name calcLikelihood
//' @param y_obs Field of vectors containing observed time points
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param nu Matrix containing current nu parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param i Int containing function number that we want the likelihood of
//' @param sigma Double containing current sigma parameter
//' @return lik Double contianing likelihood
double calcpdf(const aarma::vec& y_obs,
               const arma::mat& B_obs,
               const arma::mat& nu,
               const arma::cube& Phi,
               const arma::mat& Z,
               const arma::mat& chi,
               const int i,
               const double& sigma){
  double lik = 0;
  double mean = 0;
  for(int l = 0; l < y_obs(i,0).n_elem; l++){
    mean = 0;
    for(int k = 0; k < Z.n_cols; k++){
      if(Z(i,k) != 0){
        mean = mean + Z(i,k) * arma::dot(nu.row(k), B_obs(i,0).row(l));
        for(int n = 0; n < Phi.n_slices; n++){
          mean = mean + Z(i,k) * chi(i,n) * arma::dot(Phi.slice(n).row(k),
                          B_obs(i,0).row(l));
        }
      }
    }
    lik = R::dnorm(y_obs(i,0)(l), mean, std::sqrt(sigma), true);
  }
  return lik;
}



