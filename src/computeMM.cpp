#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

//' Computes M_i
//'
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param i Int indicating which M we are calculating
//' @param M Matrix acting as a placeholder for M
//' @export
// [[Rcpp::export]]
void computeMi(const arma::field<arma::mat>& S_obs, const arma::mat& Z,
              const arma::cube& phi, const int i, arma::mat& M)
{
  M.zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    M = M + Z(i,j) * arma::pinv(S_obs(i,1).t() * phi.slice(i) *
      phi.slice(i).t() * S_obs(i,1));
  }
  M = arma::pinv(M);
}

//' Computes all M matrices
//'
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param M Cube containing all M matrices
//' @export
// [[Rcpp::export]]
void computeM(const arma::field<arma::mat>& S_obs, const arma::mat& Z,
             const arma::cube& phi, arma::cube& M)
{
  M.zeros();
  for(int i = 0; i < Z.n_rows; i++)
  {
    for(int j = 0; j < Z.n_cols; j++)
    {
      M.slice(i) = M.slice(i) + Z(i,j) * arma::pinv(S_obs(i,1).t() * phi.slice(i) *
        phi.slice(i).t() * S_obs(i,1));
    }
    M.slice(i) = arma::pinv(M.slice(i));
  }
}

//' Computes m_i
//'
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param nu Matrix of current nu parameters
//' @param i Int indicating which M we are calculating
//' @param m vector acting as a placeholder for m
//' @export
// [[Rcpp::export]]
void compute_mi(const arma::field<arma::mat>& S_obs, const arma::mat& Z,
               const arma::cube& phi, const arma::mat& nu, const int i,
               arma::vec& m)
{
  m.zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    m = m + Z(i,j) * arma::pinv(S_obs(i,1).t() * phi.slice(i) *
      phi.slice(i).t() * S_obs(i,1))* S_obs(i,1) * nu.col(j);
  }
}

//' Computes all m vectors
//'
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param nu Matrix of current nu parameters
//' @param m matrix acting as a placeholder for m
//' @export
// [[Rcpp::export]]
void compute_m(const arma::field<arma::mat>& S_obs, const arma::mat& Z,
              const arma::cube& phi, const arma::mat& nu, arma::mat& m)
{
  m.zeros();
  for(int i = 0; i < Z.n_rows; i++)
  {
    for(int j = 0; j < Z.n_cols; j++)
    {
      m.row(i) = m.row(i) + Z(i,j) * arma::pinv(S_obs(i,1).t() * phi.slice(i) *
        phi.slice(i).t() * S_obs(i,1))* S_obs(i,1) * nu.col(j);
    }
  }

}
