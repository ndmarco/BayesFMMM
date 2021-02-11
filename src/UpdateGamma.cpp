#include <RcppArmadillo.h>
#include <cmath>
#include "computeMM.h"

//' Updates the gamma parameters
//'
//' @name updateGamma
//' @param nu double containing hyperparameter
//' @param iter int containing MCMC iteration
//' @param delta Matrix containing current values of delta
//' @param phi Cube containing current values of phi
//' @param Z matrix containing current values of class inclusion
//' @param gamma Field of cubes contianing MCMC samples for gamma
void updateGamma(const double& nu,
                 const int iter,
                 const arma::mat& delta,
                 const arma::cube& phi,
                 const arma::mat& Z,
                 arma::field<arma::cube>& gamma)
{
  double placeholder = 1;
  for(int i = 0; i < phi.n_rows; i++)
  {
    for(int l = 0; l < phi.n_slices; l++)
    {
      placeholder = 1;
      for(int j = 0; j < phi.n_cols; j++)
      {
        placeholder = placeholder * delta(j,l);
        gamma(0,iter)(i,j,l) = R::rgamma((nu + 1)/2, (nu + placeholder *
          (phi(i,j,l) * phi(i,j,l)))/2);
      }
    }
  }
}

//' Updates the gamma parameters for single covariance matrix
//'
//' @name updateGamma
//' @param nu double containing hyperparameter
//' @param iter int containing MCMC iteration
//' @param delta Vector containing current values of delta
//' @param phi Matrix containing current values of phi
//' @param gamma Field of Matrices contianing MCMC samples for gamma
void updateGamma(const double& nu,
                 const int iter,
                 const arma::vec& delta,
                 const arma::mat& phi,
                 arma::field<arma::mat>& gamma)
{
  double placeholder = 1;
  for(int i = 0; i < phi.n_rows; i++)
  {
    for(int j = 0; j < phi.n_cols; j++)
    {
      placeholder = placeholder * delta(j);
      gamma(0,iter)(i,j) = R::rgamma((nu + 1)/2, (nu + placeholder *
        (phi(i,j) * phi(i,j)))/2);
    }
  }
}
