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
//' @param mp_inv Matrix acting as a placeholder for mp-inv of covariance
//' @param tilde_M Matrix acting as a placeholder for M
void compute_tildeMi(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs,
                    const arma::mat& Z,
                    const arma::cube& phi,
                    const int i,
                    arma::mat& mp_inv,
                    arma::mat& tilde_M)
{
  int n1 = 0;
  int n2 = 0;
  tilde_M.zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    n1 = S_obs(i,0).n_rows;
    n2 = S_star(i,0).n_rows;
    mp_inv.submat(0, 0, n1 - 1, n1 - 1) = S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t();
    mp_inv.submat(0, n1, n1 - 1 , n1 + n2 - 1) =  S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
    mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv.submat(0, n1, n1 - 1 , n1 + n2 - 1).t();
    mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();

    // compute M-P inverse
    arma::pinv(mp_inv, mp_inv);
    tilde_M = tilde_M + Z(i,j) * mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1);
  }
  arma::pinv(tilde_M, tilde_M);
}

//' Computes all tilde M
//'
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param mp_inv Field of Matrices acting as a placeholder for mp-inv of covariance
//' @param tilde_M Field of Matrices acting as a placeholder for M
void compute_tildeM(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs,
                   const arma::mat& Z,
                   const arma::cube& phi,
                   arma::field<arma::mat>& mp_inv,
                   arma::field<arma::mat>& tilde_M)
{
  int n1 = 0;
  int n2 = 0;
  for(int i = 0; i < Z.n_rows; i++)
  {
    tilde_M(i,0).zeros();
    for(int j = 0; j < Z.n_cols; j++)
    {
      n1 = S_obs(i,0).n_rows;
      n2 = S_star(i,0).n_rows;
      mp_inv(i,0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t();
      mp_inv(i,0).submat(0, n1, n1 - 1 , n1 + n2 - 1) =  S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
      mp_inv(i,0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i,0).submat(0, n1, n1 - 1 , n1 + n2 - 1).t();
      mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();

      // compute M-P inverse
      arma::pinv(mp_inv(i,0), mp_inv(i,0));
      tilde_M(i,0) = tilde_M(i,0) + Z(i,j) * mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1);
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
//' @param mp_inv Matrix acting as a placeholder fo mp-inverse of covariance
//' @param tilde_m vector acting as a placeholder for tilde_m
void compute_tildemi(const arma::field<arma::mat>& S_star,
                    const arma::field<arma::mat>& S_obs,
                    const arma::vec f_obs,
                    const arma::mat& Z,
                    const arma::cube& phi,
                    const arma::mat& nu,
                    const int i,
                    arma::mat& mp_inv,
                    arma::vec& tilde_m)
{
  int n1 = 0;
  int n2 = 0;
  tilde_m.zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    n1 = S_obs(i,0).n_rows;
    n2 = S_star(i,0).n_rows;
    mp_inv.submat(0, 0, n1 - 1, n1 - 1) = S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t();
    mp_inv.submat(0, n1, n1 - 1 , n1 + n2 - 1) =  S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
    mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv.submat(0, n1, n1 - 1 , n1 + n2 - 1).t();
    mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
    // compute M-P inverse
    arma::pinv(mp_inv, mp_inv);
    tilde_m = tilde_m + Z(i,j) * (mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) * (
      S_obs(i,0) * nu.col(j) - f_obs) + mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) *
        S_star(i,0) * nu.col(j));
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
//' @param mp_inv Field of Matrices acting as a placeholder fo mp-inverse of covariance
//' @param tilde_m Field of Vectors acting as a placeholder for tilde_m
void compute_tildem(const arma::field<arma::mat>& S_star,
                   const arma::field<arma::mat>& S_obs,
                   const arma::field<arma::vec>& f_obs,
                   const arma::mat& Z,
                   const arma::cube& phi,
                   const arma::mat& nu,
                   arma::field<arma::mat>& mp_inv,
                   arma::field<arma::vec>& tilde_m)
{
  int n1 = 0;
  int n2 = 0;
  for(int i = 0; i < Z.n_rows; i++)
  {
    tilde_m(i,0).zeros();
    for(int j = 0; j < Z.n_cols; j++)
    {
      n1 = S_obs(i,0).n_rows;
      n2 = S_star(i,0).n_rows;
      mp_inv(i,0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t();
      mp_inv(i,0).submat(0, n1, n1 - 1 , n1 + n2 - 1) =  S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
      mp_inv(i,0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i,0).submat(0, n1, n1 - 1 , n1 + n2 - 1).t();
      mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
      // compute M-P inverse
      arma::pinv(mp_inv(i,0), mp_inv(i,0));
      tilde_m(i,0) = tilde_m(i,0) + Z(i,j) * (mp_inv(i,0).submat(n1, 0, n1 + n2 - 1, n1 - 1) * (
        S_obs(i,0) * nu.col(j) - f_obs(i,0)) + mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) *
          S_star(i,0) * nu.col(j));
    }
  }
}

//' Computes the ith tilde M and tilde m
//'
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param f_obs vector of current f values at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param nu Matrix of current nu paramaters
//' @param i Int corresponding to the M and m matrix to be computed
//' @param mp_inv Field of Matrices acting as a placeholder fo mp-inverse of covariance
//' @param tilde_m Field of Vectors acting as a placeholder for tilde_m
void compute_tildeMi_tildemi(const arma::field<arma::mat>& S_star,
                             const arma::field<arma::mat>& S_obs,
                             const arma::field<arma::vec>& f_obs,
                             const arma::mat& Z,
                             const arma::cube& phi,
                             const arma::mat& nu,
                             const int i,
                             arma::field<arma::mat>& mp_inv,
                             arma::field<arma::mat>& tilde_M,
                             arma::field<arma::vec>& tilde_m)
{
  int n1 = 0;
  int n2 = 0;
  tilde_m(i,0).zeros();
  tilde_M(i,0).zeros();
  for(int j = 0; j < Z.n_cols; j++)
  {
    n1 = S_obs(i,0).n_rows;
    n2 = S_star(i,0).n_rows;
    mp_inv(i,0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t();
    mp_inv(i,0).submat(0, n1, n1 - 1 , n1 + n2 - 1) =  S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
    mp_inv(i,0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i,0).submat(0, n1, n1 - 1 , n1 + n2 - 1).t();
    mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
    // compute M-P inverse
    arma::pinv(mp_inv(i,0), mp_inv(i,0));
    tilde_M(i,0) = tilde_M(i,0) + Z(i,j) * mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1);
    tilde_m(i,0) = tilde_m(i,0) + Z(i,j) * (mp_inv(i,0).submat(n1, 0, n1 + n2 - 1, n1 - 1) * (
      S_obs(i,0) * nu.col(j) - f_obs(i,0)) + mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) *
        S_star(i,0) * nu.col(j));
  }
  arma::pinv(tilde_M(i,0), tilde_M(i,0));
}

//' Computes all tilde M and tilde m
//'
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param f_obs vector of current f values at observed time points
//' @param Z Matrix of current Z parameter
//' @param phi Cube of current phi paramaters
//' @param nu Matrix of current nu paramaters
//' @param mp_inv Field of Matrices acting as a placeholder fo mp-inverse of covariance
//' @param tilde_m Field of Vectors acting as a placeholder for tilde_m
void compute_tildeM_tildem(const arma::field<arma::mat>& S_star,
                           const arma::field<arma::mat>& S_obs,
                           const arma::field<arma::vec>& f_obs,
                           const arma::mat& Z,
                           const arma::cube& phi,
                           const arma::mat& nu,
                           arma::field<arma::mat>& mp_inv,
                           arma::field<arma::mat>& tilde_M,
                           arma::field<arma::vec>& tilde_m)
{
  int n1 = 0;
  int n2 = 0;
  for(int i = 0; i < Z.n_rows; i++)
  {
    tilde_m(i,0).zeros();
    tilde_M(i,0).zeros();
    for(int j = 0; j < Z.n_cols; j++)
    {
      n1 = S_obs(i,0).n_rows;
      n2 = S_star(i,0).n_rows;
      mp_inv(i,0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_obs(i,0).t();
      mp_inv(i,0).submat(0, n1, n1 - 1 , n1 + n2 - 1) =  S_obs(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
      mp_inv(i,0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i,0).submat(0, n1, n1 - 1 , n1 + n2 - 1).t();
      mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i,0) * phi.slice(j) * phi.slice(j).t() * S_star(i,0).t();
      // compute M-P inverse
      arma::pinv(mp_inv(i,0), mp_inv(i,0));
      tilde_M(i,0) = tilde_M(i,0) + Z(i,j) * mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1);
      tilde_m(i,0) = tilde_m(i,0) + Z(i,j) * (mp_inv(i,0).submat(n1, 0, n1 + n2 - 1, n1 - 1) * (
        S_obs(i,0) * nu.col(j) - f_obs(i,0)) + mp_inv(i,0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) *
          S_star(i,0) * nu.col(j));
    }
    arma::pinv(tilde_M(i,0), tilde_M(i,0));
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

