#include <RcppArmadillo.h>
#include <cmath>

//' Updates the Y_star parameters
//'
//' @name updateY_star
//' @param B_star Field of matrices containing basis functions evaluated at unobserved time points
//' @param nu Matrix contianing current nu parameters
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
                  double& sigma,
                  const int& iter,
                  const int& tot_mcmc_iters,
                  arma::field<arma::mat>& y_star){
  double mean = 0;
  for(int i = 0; i < Z.n_rows; i++){
    if(B_star(i,0).n_elem > 0){
      for(int l = 0; l < B_star(i,0).n_cols; l++){
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
        y_star(i,0)(iter, l) = R::rnorm(mean, sigma);
      }
    }
    if(iter < (tot_mcmc_iters - 1)){
      y_star(i,0).row(iter + 1) = y_star(i,0).row(iter);
    }
  }
}
