#include <RcppArmadillo.h>
#include <cmath>

//' Updates the delta parameters for individualized covariance matrix
//'
//' @name updateDelta
//' @param phi cube containing the current values of phi
//' @param gamma Cube containing current values of gamma
//' @param a Matrix containing current values of a
//' @param iter Int containing MCMC iteration number
//' @param delta Cube containing values of delta
void updateDelta(const arma::cube& phi,
                 const arma::cube& gamma,
                 const arma::mat& a,
                 const int iter,
                 arma::cube& delta)
{
  double param1 = 0;
  double param2 = 0;
  double tilde_tau = 0;
  for(int i = 0; i < phi.n_cols; i++)
  {
    for(int j = 0; j < phi.n_slices; j++)
    {
      if(i == 0)
      {
        param1 = a(1,j) + (phi.n_cols * phi.n_rows) / 2;
        param2 = 1;
        for(int k = 0; k < phi.n_rows; k++)
        {
          param2 = param2 + (0.5 * gamma(k, 1, j) * std::pow(phi(k, 1, j), 2));
          for(int m = 1; m < phi.n_rows; m++)
          {
            tilde_tau = 1;
            for(int n = 1; n <= m; n ++)
            {
              tilde_tau = tilde_tau * delta(n, j, iter);
            }
            param2 = param2 + (0.5 * gamma(k, m, j) * tilde_tau * std::pow(phi(k, m, j), 2));
          }
        }
        delta(i, j, iter) = R::rgamma(param1, param2);
      }else
      {
        param1 = a(2,j) + (phi.n_cols * (phi.n_rows - i)) / 2;
        param2 = 1;
        for(int k = 0; k < phi.n_rows; k++)
        {
          for(int m = i; m < phi.n_rows; m++)
          {
            tilde_tau = 1;
            for(int n = 1; n <= m; n ++)
            {
              if(n != i)
              {
                tilde_tau = tilde_tau * delta(n, j, iter);
              }
            }
            param2 = param2 + (0.5 * gamma(k, m, j) * tilde_tau * std::pow(phi(k, m, j), 2));
          }
        }
        delta(i, j, iter) = R::rgamma(param1, param2);
      }
    }
  }
}

//' Updates the delta parameters for single covariance matrix
//'
//' @name updateDelta
//' @param phi matrix containing the current values of phi
//' @param gamma matrix containing current values of gamma
//' @param a vector containing current values of a
//' @param iter Int containing MCMC iteration number
//' @param delta Cube containing values of delta
void updateDelta(const arma::mat& phi,
                 const arma::mat& gamma,
                 const arma::vec& a,
                 const int iter,
                 arma::mat& delta)
{
  double param1 = 0;
  double param2 = 0;
  double tilde_tau = 0;
  for(int i = 0; i < phi.n_cols; i++)
  {
    if(i == 0)
    {
      param1 = a(1) + (phi.n_cols * phi.n_rows) / 2;
      param2 = 1;
      for(int k = 0; k < phi.n_rows; k++)
      {
        param2 = param2 + (0.5 * gamma(k, 1) * std::pow(phi(k, 1), 2));
        for(int m = 1; m < phi.n_rows; m++)
        {
          tilde_tau = 1;
          for(int n = 1; n <= m; n ++)
          {
            tilde_tau = tilde_tau * delta(n, iter);
          }
          param2 = param2 + (0.5 * gamma(k, m) * tilde_tau * std::pow(phi(k, m), 2));
        }
      }
      delta(i, iter) = R::rgamma(param1, param2);
    }else
    {
      param1 = a(2) + (phi.n_cols * (phi.n_rows - i)) / 2;
      param2 = 1;
      for(int k = 0; k < phi.n_rows; k++)
      {
        for(int m = i; m < phi.n_rows; m++)
        {
          tilde_tau = 1;
          for(int n = 1; n <= m; n ++)
          {
            if(n != i)
            {
              tilde_tau = tilde_tau * delta(n, iter);
            }
          }
          param2 = param2 + (0.5 * gamma(k, m) * tilde_tau * std::pow(phi(k, m), 2));
        }
      }
      delta(i, iter) = R::rgamma(param1, param2);
    }
  }
}
