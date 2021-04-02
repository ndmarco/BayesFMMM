#include <RcppArmadillo.h>
#include <cmath>

//' Updates the Tau parameters
//'
//' @name updateTau
//' @param alpha Double containing hyperparameter
//' @param beta Double containing hyperparameter
//' @param nu Matrix contianing nu parameters
//' @param iter Int containing current MCMC iteration
//' @param tot_mcmc_iters Int containing total number of MCMC iterations
//' @param P Matrix that acts as a placeholder for P
//' @param tau Matrix containing tau for all mcmc iterations
void updateSigma(const arma::field<arma::vec>& y_obs,
                 const arma::field<arma::mat>& y_star,
                 const arma::field<arma::mat>& B_obs,
                 const arma::field<arma::mat>& B_star,
                 const double alpha_0,
                 const double beta_0,
                 const arma::mat& nu,
                 const arma::cube& Phi,
                 const arma::mat& Z,
                 const arma::mat& chi,
                 const int& iter,
                 const int& tot_mcmc_iters,
                 arma::vec& sigma){
  double a = 0;
  double b = 0;
  double b_1 = 0;
  for(int i = 0; i < Z.n_rows; i++){
    for(int l = 0; l < y_obs(i,0).n_elem; l++){
      b = y_obs(i,0)(l);
      for(int k = 0; k < Z.n_cols; k++){
        if(Z(i,k) != 0){
          b = b - arma::dot(nu.row(k), B_obs(i,0).row(l));
          for(int n = 0; n < Phi.n_slices; n++){
            b = b - chi(i,n) * arma::dot(Phi.slice(n).row(k), B_obs(i,0).row(l));
          }
        }
      }
      b_1 = b_1 + 0.5 * (b * b);
    }
    a = a + (y_obs(i,0).n_elem / 2);
    if(B_star(i,0).n_elem > 0){
      for(int l = 0; l < y_star(i,0).n_cols; l++){
        b = y_star(i,0)(iter, l);
        for(int k = 0; k < Z.n_cols; k++){
          if(Z(i,k) != 0){
            b = b - arma::dot(nu.row(k), B_star(i,0).row(l));
            for(int n = 0; n < Phi.n_slices; n++){
              b = b - chi(i,n) * arma::dot(Phi.slice(n).row(k),
                          B_star(i,0).row(l));
            }
          }
        }
        b_1 = b_1 + 0.5 * (b * b);
      }
      a = a + (y_star(i,0).n_cols / 2);
    }
  }
  b_1 = b_1 + beta_0;
  a = a + alpha_0;
  sigma(iter) = 1 / R::rgamma(a, 1/b_1);

  if(iter < (tot_mcmc_iters - 1)){
    sigma(iter + 1) = sigma(iter);
  }
}
