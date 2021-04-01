#include <RcppArmadillo.h>
#include <cmath>
#include "Distributions.H"

//' Updates the gamma parameters
//'
//' @name updateGamma
//' @param nu double containing hyperparameter
//' @param iter int containing MCMC iteration
//' @param delta Matrix containing current values of delta
//' @param phi Cube containing current values of phi
//' @param Z matrix containing current values of class inclusion
//' @param gamma Field of cubes contianing MCMC samples for gamma
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
              arma::vec& b_1,
              arma::mat& B_1,
              arma::mat& P,
              arma::cube& nu){
  double ph = 0;
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
    P.zeros();
    for(int j = 0; j < nu.n_rows; j++){
      P(0,0) = 1;
      P(nu.n_rows - 1, nu.n_rows - 1) = 1;
      if(j > 0){
        P(j,j) = 2;
        P(j-1,j) = -1;
        P(j,j-1) = -1;
      }
    }
    B_1 = B_1 + tau(j) * P;
    arma::inv(B_1, B_1);
    nu.slice(iter).row(j) = Rmvnormal(B_1 * b_1, B_1).t();
  }
  if(iter < (tot_mcmc_iters - 1)){
    nu.slice(iter + 1) = nu.slice(iter);
  }
}
