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
void computeMi(const arma::field<arma::mat>& S_obs,
               const arma::mat& Z,
               const arma::cube& phi,
               const int i,
               arma::mat& M)
{
  M.zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    M = M + Z(i,j) * arma::pinv(S_obs(i,0) * phi.slice(j) *
      phi.slice(j).t() * S_obs(i,0).t());
  }
  M = arma::pinv(M);
}

//' Computes all M matrices
//'
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param M Field of Matrices containing all M matrices
void computeM(const arma::field<arma::mat>& S_obs,
              const arma::mat& Z,
              const arma::cube& phi,
              arma::field<arma::mat>& M)
{
  for(int i = 0; i < Z.n_rows; i++)
  {
    M(i,0).zeros();
    for(int j = 0; j < Z.n_cols; j++)
    {
      M(i,0) = M(i,0) + Z(i,j) * arma::pinv(S_obs(i,0) * phi.slice(j) *
        phi.slice(j).t() * S_obs(i,0).t());
    }
    arma::pinv(M(i,0), M(i,0));
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
void compute_mi(const arma::field<arma::mat>& S_obs,
                const arma::mat& Z,
                const arma::cube& phi,
                const arma::mat& nu,
                const int i,
                arma::vec& m)
{
  m.zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    m = m + Z(i,j) * arma::pinv(S_obs(i,0) * phi.slice(j) *
      phi.slice(j).t() * S_obs(i,0).t())* S_obs(i,0) * nu.col(j);
  }
}

//' Computes all m vectors
//'
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param nu Matrix of current nu parameters
//' @param m Field of vectors acting as a placeholder for m
void compute_m(const arma::field<arma::mat>& S_obs,
               const arma::mat& Z,
               const arma::cube& phi,
               const arma::mat& nu,
               arma::field<arma::vec>& m)
{
  for(int i = 0; i < Z.n_rows; i++)
  {
    m(i,0).zeros();
    for(int j = 0; j < Z.n_cols; j++)
    {
      m(i,0) = m(i,0) + Z(i,j) * arma::pinv(S_obs(i,0) * phi.slice(j) *
        phi.slice(j).t() * S_obs(i,0).t())* S_obs(i,0) * nu.col(j);
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
void compute_tildeMi(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs,
                    const arma::mat& Z,
                    const arma::cube& phi,
                    const int i,
                    arma::mat& Z_plus,
                    arma::mat& M)
{
  M.zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    Z_plus = (S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t()) -
      ((S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t()) *
      arma::pinv(S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t()) *
      (S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t()));
    // compute M-P inverse
    arma::pinv(Z_plus, Z_plus);
    M = M + Z(i,j) * Z_plus;
  }
  arma::pinv(M, M);
}

//' Computes all tilde M
//'
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param Z_plus Field of Matrices acting as a placeholder for Z_plus.
//' @param tilde_M Field of Matrices acting as a placeholder for M
void compute_tildeM(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs,
                   const arma::mat& Z,
                   const arma::cube& phi,
                   arma::field<arma::mat>& Z_plus,
                   arma::field<arma::mat>& tilde_M)
{
  for(int i = 0; i < Z.n_rows; i++)
  {
    tilde_M(i,0).zeros();
    for(int j = 0; j < Z.n_cols; j++)
    {
      Z_plus(i,0) = (S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t()) -
        ((S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t()) *
        arma::pinv(S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t()) *
        (S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t()));
      // compute M-P inverse
      arma::pinv(Z_plus(i,0), Z_plus(i,0));
      tilde_M(i,0) = tilde_M(i,0) + Z(i,j) * Z_plus(i,0);
    }
    arma::pinv(tilde_M(i,0), tilde_M(i,0));
  }
}

//' Computes tilde m_i
//'
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param f_obs vector of current f values at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param nu Matrix of current nu paramaters
//' @param i Int indicating which tilde_m we are calculating
//' @param Z_plus Matrix acting as a placeholder for Z_plus
//' @param A_plus Matrix acting as a placeholder for A_plus
//' @param C Matrix acting as a placeholder for C
//' @param tilde_m vector acting as a placeholder for tilde_m
void compute_tildemi(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs,
                    const arma::vec f_obs,
                    const arma::mat& Z,
                    const arma::cube& phi,
                    const arma::mat nu,
                    const int i,
                    arma::mat& Z_plus,
                    arma::mat& A_plus,
                    arma::mat& C,
                    arma::vec& tilde_m)
{
  tilde_m.zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    C = S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t();
    A_plus = arma::pinv(S_obs(i,0) * phi.slice(j) * phi.slice(j).t() *
      S_obs(i,0).t());
    Z_plus = (S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t()) -
      (C * A_plus * (S_obs(i,0) * phi.slice(j) * phi.slice(j).t() *
      S_star(i,0).t()));
    // compute M-P inverse
    arma::pinv(Z_plus, Z_plus);
    tilde_m = tilde_m + Z(i,j) * Z_plus * (C * A_plus * f_obs + S_star(i,0) *
      nu.col(j));
  }
}

//' Computes all tilde m
//'
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param f_obs vector of current f values at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param nu Matrix of current nu paramaters
//' @param Z_plus Field of Matrices acting as a placeholder for Z_plus
//' @param A_plus Field of Matrices acting as a placeholder for A_plus
//' @param C Field of Matrices acting as a placeholder for C
//' @param tilde_m Field of Vectors acting as a placeholder for tilde_m
void compute_tildem(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs,
                   const arma::field<arma::vec>& f_obs,
                   const arma::mat& Z,
                   const arma::cube& phi,
                   const arma::mat nu,
                   arma::field<arma::mat>& Z_plus,
                   arma::field<arma::mat>& A_plus,
                   arma::field<arma::mat>& C,
                   arma::field<arma::vec>& tilde_m)
{
  for(int i = 0; i < Z.n_rows; i++)
  {
    tilde_m(i,0).zeros();
    for(int j = 0; j < Z.n_cols; j++)
    {
      C(i,0) = S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t();
      A_plus(i,0) = arma::pinv(S_obs(i,0) * phi.slice(j) * phi.slice(j).t() *
        S_obs(i,0).t());
      Z_plus(i,0) = (S_star(i,0) * phi.slice(j) * phi.slice(j).t() *
        S_star(i,0).t()) - (C(i,0) * A_plus(i,0) * (S_obs(i,0) * phi.slice(j) *
        phi.slice(j).t() * S_star(i,0).t()));
      // compute M-P inverse
      arma::pinv(Z_plus(i,0), Z_plus(i,0));
      tilde_m(i,0) = tilde_m(i,0) + Z(i,j) * Z_plus(i,0) * (C(i,0) *
        A_plus(i,0) * f_obs(i,0) + S_star(i,0) * nu.col(j));
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
