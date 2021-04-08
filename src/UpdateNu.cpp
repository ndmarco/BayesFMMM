#include <RcppArmadillo.h>
#include <cmath>
#include "Distributions.H"

//' Updates the nu parameters
//'
//' @name updateNu
//' @param y_obs Field of vectors containing observed time points
//' @param y_obs Field of matrices containing unobserved time points for all iterations
//' @param B_obs Field of matrices containing basis functions evaluated at observed time points
//' @param B_star Field of matrices containing basis functions evaluated at unobserved time points
//' @param tau Vector containing current tau parameters
//' @param Phi Cube containing current Phi parameters
//' @param Z Matrix containing current Z parameters
//' @param chi Matrix containing current chi parameters
//' @param sigma Double containing current sigma parameter
//' @param iter Int containing MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param P Matrix contiaing tridiagonal P matrix
//' @param b_1 Vector acting as a placeholder for mean vector
//' @param B_1 Matrix acting as placeholder for covariance matrix
//' @param nu Cube contianing MCMC samples for nu
void updateNu(const arma::field<arma::vec>& y_obs,
              const arma::field<arma::mat>& y_star,
              const arma::field<arma::mat>& B_obs,
              const arma::field<arma::mat>& B_star,
              const arma::vec& tau,
              const arma::cube& Phi,
              const arma::mat& Z,
              const arma::mat& chi,
              const double& sigma,
              const int& iter,
              const int& tot_mcmc_iters,
              const arma::mat& P,
              arma::vec& b_1,
              arma::mat& B_1,
              arma::cube& nu){
  double ph = 0;
  // initialize P matrix
  for(int j = 0; j < nu.n_rows; j++){
    b_1.zeros();
    B_1.zeros();
    for(int i = 0; i < Z.n_rows; i++){
      if(Z(i,j) != 0){
        for(int l = 0; l < y_obs(i,0).n_elem; l++){
          ph = 0;
          ph = y_obs(i,0)(l);
          B_1 = B_1 + (B_obs(i,0).row(l).t() * B_obs(i,0).row(l));
          for(int k = 0; k < nu.n_rows; k++){
            if(Z(i,k) != 0){
              if(k != j){
                ph = ph -  (arma::dot(nu.slice(iter).row(k),
                             B_obs(i,0).row(l)));
              }
              for(int n = 0; n < Phi.n_slices; n++){
                ph = ph - (chi(i,n) * arma::dot(Phi.slice(n).row(k),
                               B_obs(i,0).row(l)));
              }
            }
          }
          b_1 = b_1 + B_obs(i,0).row(l).t() * ph;
        }
        if(B_star(i,0).n_elem > 0){
          for(int l = 0; l < y_star(i,0).n_cols; l++){
            ph = 0;
            ph = y_star(i,0)(iter, l);
            B_1 = B_1 + (B_star(i,0).row(l).t() * B_star(i,0).row(l));
            for(int k = 0; k < nu.n_rows; k++){
              if(Z(i,k) != 0){
                if(k != j){
                  ph = ph -  (arma::dot(nu.slice(iter).row(k),
                                        B_star(i,0).row(l)));
                }
                for(int n = 0; n < Phi.n_slices; n++){
                  ph = ph - (chi(i,n) * arma::dot(Phi.slice(n).row(k),
                                 B_star(i,0).row(l)));
                }
              }
            }
            b_1 = b_1 + B_star(i,0).row(l).t() * ph;
          }
        }
      }
    }
    b_1 = b_1 / sigma;
    B_1 = B_1 / sigma;
    B_1 = B_1 + tau(j) * P;
    arma::inv(B_1, B_1);
    nu.slice(iter).row(j) = arma::mvnrnd(B_1 * b_1, B_1).t();
  }
  if(iter < (tot_mcmc_iters - 1)){
    nu.slice(iter + 1) = nu.slice(iter);
  }
}
