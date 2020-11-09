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
    M = M + Z(i,j) * arma::pinv(S_obs(i,1) * phi.slice(j) *
      phi.slice(j).t() * S_obs(i,1).t());
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
      M.slice(i) = M.slice(i) + Z(i,j) * arma::pinv(S_obs(i,1) * phi.slice(j) *
        phi.slice(j).t() * S_obs(i,1).t());
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
    m = m + Z(i,j) * arma::pinv(S_obs(i,1) * phi.slice(j) *
      phi.slice(j).t() * S_obs(i,1).t())* S_obs(i,1).t() * nu.col(j);
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
      m.row(i) = m.row(i) + Z(i,j) * arma::pinv(S_obs(i,1) * phi.slice(j) *
        phi.slice(j).t() * S_obs(i,1).t())* S_obs(i,1).t() * nu.col(j);
    }
  }

}

//' Computes tilde M_i
//'
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param i Int indicating which M we are calculating
//' @param Z_plus Matrix acting as a placeholder for Z_plus.
//' @param M Matrix acting as a placeholder for M
void compute_tileMi(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs, const arma::mat& Z,
                   const arma::cube& phi, const int i, arma::mat& Z_plus,
                   arma::mat& M)
{
  M.zeros();
  for(int j = 0; j < Z.n_cols; j ++)
  {
    Z_plus = (S_star(i,1) * phi.slice(j) * phi.slice(j).t() * S_star(i,1).t()) -
      ((S_star(i,1) * phi.slice(j) * phi.slice(j).t() * S_obs(i,1).t()) *
      arma::pinv(S_obs(i,1) * phi.slice(j) * phi.slice(j).t() * S_obs(i,1).t()) *
      (S_obs(i,1) * phi.slice(j) * phi.slice(j).t() * S_star(i,1).t()));
    // compute M-P inverse
    arma::pinv(Z_plus, Z_plus);
    M = M + Z(i,j) * Z_plus;
  }
}

//' Computes all tilde M
//'
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param i Int indicating which M we are calculating
//' @param Z_plus Field of Matrices acting as a placeholder for Z_plus.
//' @param M Field of Matrices acting as a placeholder for M
void compute_tileM(const arma::field<arma::mat>& S_star,
                  const arma::field<arma::mat>& S_obs, const arma::mat& Z,
                  const arma::cube& phi, arma::field<arma::mat>& Z_plus,
                  arma::field<arma::mat>& M)
{
  for(int i = 0; i < Z.n_rows; i++)
  {
    M(i,1).zeros();
    for(int j = 0; j < Z.n_cols; j ++)
    {
      Z_plus(i,1) = (S_star(i,1) * phi.slice(j) * phi.slice(j).t() * S_star(i,1).t()) -
        ((S_star(i,1) * phi.slice(j) * phi.slice(j).t() * S_obs(i,1).t()) *
        arma::pinv(S_obs(i,1) * phi.slice(j) * phi.slice(j).t() * S_obs(i,1).t()) *
        (S_obs(i,1) * phi.slice(j) * phi.slice(j).t() * S_star(i,1).t()));
      // compute M-P inverse
      arma::pinv(Z_plus(i,1), Z_plus(i,1));
      M(i,1) = M(i,1) + Z(i,j) * Z_plus(i,1);
    }
  }
}

// //' sample p_inv
// //'
// //' @param M Matrix to be inverted
// //' @export
// // [[Rcpp::export]]
// void pinv_arma(arma::mat& M)
// {
//   arma::pinv(M,M);
// }
//
// //' @export
// // [[Rcpp::export]]
// Rcpp::List call_it(){
//   arma::cube X(5, 5, 2, arma::fill::randu);
//   arma::mat T(4,5, arma::fill::randu);
//   X.slice(0) = T.t() * T;
//   arma::mat S = X.slice(0);
//   pinv_arma(X.slice(0));
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("X1", X.slice(0)),
//                                       Rcpp::Named("X2", X.slice(1)),
//                                       Rcpp::Named("S", S));
//   return mod;
// }
