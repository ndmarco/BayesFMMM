#include <RcppArmadillo.h>
#include <cmath>
#include "CalculateCov.H"

// [[Rcpp::depends(RcppArmadillo)]]

//' Computes M_i
//'
//' @name compute_Mi
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Cube of current Phi paramaters
//' @param Rho Matrix with each row containing the elements of the upper triangular matrix
//' @param i Int indicating which M we are calculating
//' @param Cov Matrix containing placeholder for covariance matrix
//' @param mp_inv Matrix acting as a placeholder for mp inverse
//' @param M Matrix acting as a placeholder for M

void compute_Mi(const arma::field<arma::mat>& S_obs,
                const arma::field<arma::mat>& S_star,
                const arma::mat& Z,
                const arma::cube& Phi,
                const arma::mat& Rho,
                const int i,
                arma::mat& Cov,
                arma::mat& mp_inv,
                arma::mat& M)
{
  M.zeros();
  int n1 = S_obs(i, 0).n_rows;
  int n2 = S_star(i, 0).n_rows;

  getCov(Z.row(i), Phi, Rho, Cov);
  mp_inv.submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Cov *
    S_obs(i, 0).t();
  mp_inv.submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Cov *
    S_star(i, 0).t();
  mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv.submat(0, n1, n1 - 1,
                n1 + n2 - 1).t();
  mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Cov *
    S_star(i, 0).t();

  // Get full precision matrix of observed and unobserved time points
  arma::pinv(mp_inv, mp_inv);

  // Get covariance matrix of unobserved data
  arma::pinv(M, mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1));
}

//' Computes M_i for single covariance matrix
//'
//' @name compute_Mi
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Z Matrix of current Z parameter
//' @param Phi Matrix of current Phi paramaters
//' @param i Int indicating which M we are calculating
//' @param M Matrix acting as a placeholder for M

void compute_Mi(const arma::field<arma::mat>& S_obs,
                const arma::field<arma::mat>& S_star,
                const arma::mat& Z,
                const arma::mat& Phi,
                const int i,
                arma::mat& mp_inv,
                arma::mat& M)
{
  M.zeros();
  int n1 = S_obs(i, 0).n_rows;
  int n2 = S_star(i, 0).n_rows;

  mp_inv.submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Phi * Phi.t() *
    S_obs(i, 0).t();
  mp_inv.submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Phi * Phi.t() *
    S_star(i, 0).t();
  mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv.submat(0, n1, n1 - 1,
                n1 + n2 - 1).t();
  mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Phi * Phi.t() *
    S_star(i, 0).t();

  // Get full precision matrix of observed and unobserved time points
  arma::pinv(mp_inv, mp_inv);

  // Get covariance matrix of unobserved data
  arma::pinv(M, mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1));
  arma::pinv(M, M);
}

//' Computes all M matrices
//'
//' @name compute_M
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Cube of current Phi paramaters
//' @param Rho Matrix with each row containing the elements of the upper triangular matrix
//' @param Cov Matrix containing placeholder for covariance matrix
//' @param mp_inv Field of Matrices acting as a placeholder for mp inverse
//' @param M Field of Matrices acting as a placeholder for M

void compute_M(const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::mat& Z,
               const arma::cube& Phi,
               const arma::mat& Rho,
               arma::mat& Cov,
               arma::field<arma::mat>& mp_inv,
               arma::field<arma::mat>& M)
{
  int n1 = 0;
  int n2 = 0;
  for(int i = 0; i < Z.n_rows; i++)
  {
    M(i, 0).zeros();
    n1 = S_obs(i, 0).n_rows;
    n2 = S_star(i, 0).n_rows;

    getCov(Z.row(i), Phi, Rho, Cov);
    mp_inv(i, 0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Cov *
      S_obs(i, 0).t();
    mp_inv(i, 0).submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Cov *
      S_star(i, 0).t();
    mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i, 0).submat(0, n1,
           n1 - 1, n1 + n2 - 1).t();
    mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Cov *
      Cov.t() * S_star(i, 0).t();

    // Get full precision matrix of observed and unobserved time points
    arma::pinv(mp_inv(i, 0), mp_inv(i, 0));

    // Get covariance matrix of unobserved data
    arma::pinv(M(i, 0), mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1));
  }
}


//' Computes all M matrices
//'
//' @name compute_M
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Matrix of current Phi paramaters
//' @param i Int indicating which M we are calculating
//' @param mp_inv Field of Matrices acting as a placeholder for mp inverse
//' @param M Field of Matrices acting as a placeholder for M

void compute_M(const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::mat& Z,
               const arma::mat& Phi,
               arma::field<arma::mat>& mp_inv,
               arma::field<arma::mat>& M)
{
  int n1 = 0;
  int n2 = 0;
  for(int i = 0; i < Z.n_rows; i++)
  {
    M(i, 0).zeros();
    n1 = S_obs(i, 0).n_rows;
    n2 = S_star(i, 0).n_rows;

    mp_inv(i, 0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Phi * Phi.t() *
      S_obs(i, 0).t();
    mp_inv(i, 0).submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Phi *
      Phi.t() * S_star(i, 0).t();
    mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i, 0).submat(0, n1,
           n1 - 1, n1 + n2 - 1).t();
    mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) *
      Phi * Phi.t() * S_star(i, 0).t();

    // Get full precision matrix of observed and unobserved time points
    arma::pinv(mp_inv(i, 0), mp_inv(i, 0));

    // Get covariance matrix of unobserved data
    arma::pinv(M(i, 0), mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1));
  }
}

//' Computes m_i
//'
//' @name compute_mi
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Cube of current Phi paramaters
//' @param Rho Matrix with each row containing the elements of the upper triangular matrix
//' @param nu Matrix that contains mean vectors as columns
//' @param i Int indicating which M we are calculating
//' @param Cov Matrix containing placeholder for covariance matrix
//' @param mp_inv Matrix acting as a placeholder for mp inverse
//' @param mean_ph_obs vector acting as placeholder for mean of f_i at observed time points
//' @param mean_ph_star vector acting as placeholder for vector with length of unobserved time points
//' @param m Matrix acting as a placeholder for m

void compute_mi(const arma::field<arma::mat>& S_obs,
                const arma::field<arma::mat>& S_star,
                const arma::field<arma::vec>& f_obs,
                const arma::mat& Z,
                const arma::cube& Phi,
                const arma::mat& Rho,
                const arma::mat& nu,
                const int i,
                arma::mat& Cov,
                arma::mat& mp_inv,
                arma::vec& mean_ph_obs,
                arma::vec& mean_ph_star,
                arma::vec& m)
{
  m.zeros();
  int n1 = S_obs(i, 0).n_rows;
  int n2 = S_star(i, 0).n_rows;

  getCov(Z.row(i), Phi, Rho, Cov);
  mp_inv.submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Cov *
    S_obs(i, 0).t();
  mp_inv.submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Cov *
    S_star(i, 0).t();
  mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv.submat(0, n1, n1 - 1,
                n1 + n2 - 1).t();
  mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Cov *
    S_star(i, 0).t();

  // Get full precision matrix of observed and unobserved time points
  arma::pinv(mp_inv, mp_inv);

  mean_ph_obs.zeros();
  mean_ph_star.zeros();
  for(int j = 0; j < nu.n_cols; j ++){
    mean_ph_obs = mean_ph_obs + Z(i,j) * S_obs(i, 0) * nu.col(j);
    mean_ph_star = mean_ph_star + Z(i,j) * S_star(i, 0) * nu.col(j);
  }
  // Compute mean
  m = mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs - f_obs(i, 0)) +
    mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star;
}

//' Computes m_i under common covariance matrix
//'
//' @name compute_mi
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Matrix of current Phi paramaters
//' @param nu Matrix containing mean vectors as the columns
//' @param i Int indicating which M we are calculating
//' @param mp_inv Matrix acting as a placeholder for mp inverse
//' @param mean_ph_obs vector acting as placeholder for mean of f_i at observed time points
//' @param mean_ph_star vector acting as placeholder for vector with length of unobserved time points
//' @param m Matrix acting as a placeholder for m

void compute_mi(const arma::field<arma::mat>& S_obs,
                const arma::field<arma::mat>& S_star,
                const arma::field<arma::vec>& f_obs,
                const arma::mat& Z,
                const arma::mat& Phi,
                const arma::mat& nu,
                const int i,
                arma::mat& mp_inv,
                arma::vec& mean_ph_obs,
                arma::vec& mean_ph_star,
                arma::vec& m)
{
  m.zeros();
  int n1 = S_obs(i, 0).n_rows;
  int n2 = S_star(i, 0).n_rows;

  mp_inv.submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Phi *
    Phi.t() * S_obs(i, 0).t();
  mp_inv.submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Phi *
    Phi.t() * S_star(i, 0).t();
  mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv.submat(0, n1, n1 - 1,
                n1 + n2 - 1).t();
  mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Phi *
    Phi.t() * S_star(i, 0).t();

  // Get full precision matrix of observed and unobserved time points
  arma::pinv(mp_inv, mp_inv);

  mean_ph_obs.zeros();
  mean_ph_star.zeros();
  for(int j = 0; j < nu.n_cols; j ++){
    mean_ph_obs = mean_ph_obs + Z(i,j) * S_obs(i, 0) * nu.col(j);
    mean_ph_star = mean_ph_star + Z(i,j) * S_star(i, 0) * nu.col(j);
  }
  // Compute mean
  m = mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs - f_obs(i, 0)) +
    mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star;
}

//' Computes m for all observations
//'
//' @name compute_m
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Cube of current Phi paramaters
//' @param Rho Matrix with each row containing the elements of the upper triangular matrix
//' @param nu Matrix that contains mean vectors as columns
//' @param Cov Matrix containing placeholder for covariance matrix
//' @param mp_inv Field of Matrices acting as a placeholder for mp inverse
//' @param mean_ph_obs Field of Vectors acting as placeholder for mean of f_i at observed time points
//' @param mean_ph_star Field of Vectors acting as placeholder for vector with length of unobserved time points
//' @param m Field of Vectors acting as a placeholder for m

void compute_m(const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::field<arma::vec>& f_obs,
               const arma::mat& Z,
               const arma::cube& Phi,
               const arma::mat& Rho,
               const arma::mat& nu,
               arma::mat& Cov,
               arma::field<arma::mat>& mp_inv,
               arma::field<arma::vec>& mean_ph_obs,
               arma::field<arma::vec>& mean_ph_star,
               arma::field<arma::vec>& m)
{
  for(int i =0; i < Z.n_rows; i++)
  {
    m(i, 0).zeros();
    int n1 = S_obs(i, 0).n_rows;
    int n2 = S_star(i, 0).n_rows;

    getCov(Z.row(i), Phi, Rho, Cov);
    mp_inv(i, 0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Cov *
      S_obs(i, 0).t();
    mp_inv(i, 0).submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Cov *
      S_star(i, 0).t();
    mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i, 0).submat(0, n1, n1 - 1,
                  n1 + n2 - 1).t();
    mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Cov *
      S_star(i, 0).t();

    // Get full precision matrix of observed and unobserved time points
    arma::pinv(mp_inv(i, 0), mp_inv(i, 0));

    mean_ph_obs(i, 0).zeros();
    mean_ph_star(i, 0).zeros();
    for(int j = 0; j < nu.n_cols; j ++){
      mean_ph_obs(i, 0) = mean_ph_obs(i, 0) + Z(i,j) * S_obs(i, 0) * nu.col(j);
      mean_ph_star(i, 0) = mean_ph_star(i, 0) + Z(i,j) * S_star(i, 0) *
        nu.col(j);
    }
    // Compute mean
    m(i, 0) = mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs(i, 0)
                                                                 - f_obs(i, 0)) +
      mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star(i, 0);
  }
}

//' Computes m for all observations under common variance model
//'
//' @name compute_m
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Matrix of current Phi paramaters
//' @param nu Matrix containing the mean vectors as columns
//' @param mp_inv Field of Matrices acting as a placeholder for mp inverse
//' @param mean_ph_obs Field of Vectors acting as placeholder for mean of f_i at observed time points
//' @param mean_ph_star Field of Vectors acting as placeholder for vector with length of unobserved time points
//' @param m Field of Vectors acting as a placeholder for m

void compute_m(const arma::field<arma::mat>& S_obs,
               const arma::field<arma::mat>& S_star,
               const arma::field<arma::vec>& f_obs,
               const arma::mat& Z,
               const arma::mat& Phi,
               const arma::mat& nu,
               arma::field<arma::mat>& mp_inv,
               arma::field<arma::vec>& mean_ph_obs,
               arma::field<arma::vec>& mean_ph_star,
               arma::field<arma::vec>& m)
{
  for(int i =0; i < Z.n_rows; i++)
  {
    m(i, 0).zeros();
    int n1 = S_obs(i, 0).n_rows;
    int n2 = S_star(i, 0).n_rows;

    mp_inv(i, 0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Phi *
      Phi.t() * S_obs(i, 0).t();
    mp_inv(i, 0).submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Phi *
      Phi.t() * S_star(i, 0).t();
    mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i, 0).submat(0, n1,
           n1 - 1, n1 + n2 - 1).t();
    mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) *
      Phi * Phi.t() * S_star(i, 0).t();

    // Get full precision matrix of observed and unobserved time points
    arma::pinv(mp_inv(i, 0), mp_inv(i, 0));

    mean_ph_obs(i, 0).zeros();
    mean_ph_star(i, 0).zeros();
    for(int j = 0; j < nu.n_cols; j ++){
      mean_ph_obs(i, 0) = mean_ph_obs(i, 0) + Z(i,j) * S_obs(i, 0) * nu.col(j);
      mean_ph_star(i, 0) = mean_ph_star(i, 0) + Z(i,j) * S_star(i, 0) *
        nu.col(j);
    }
    // Compute mean
    m(i, 0) = mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs(i, 0)
                                                                 - f_obs(i, 0)) +
      mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star(i, 0);
  }
}

//' Computes M and m for all observations
//'
//' @name compute_M_m
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Cube of current Phi paramaters
//' @param Rho Matrix with each row containing the elements of the upper triangular matrix
//' @param nu Matrix that contains mean vectors as columns
//' @param Cov Matrix containing placeholder for covariance matrix
//' @param mp_inv Matrix acting as a placeholder for mp inverse
//' @param mean_ph_obs vector acting as placeholder for mean of f_i at observed time points
//' @param mean_ph_star vector acting as placeholder for vector with length of unobserved time points
//' @param M Matrix acting as a placeholder for M

void compute_M_m(const arma::field<arma::mat>& S_obs,
                 const arma::field<arma::mat>& S_star,
                 const arma::field<arma::vec>& f_obs,
                 const arma::mat& Z,
                 const arma::cube& Phi,
                 const arma::mat& Rho,
                 const arma::mat& nu,
                 arma::mat& Cov,
                 arma::field<arma::mat>& mp_inv,
                 arma::field<arma::vec>& mean_ph_obs,
                 arma::field<arma::vec>& mean_ph_star,
                 arma::field<arma::vec>& m,
                 arma::field<arma::mat>& M)
{
  for(int i =0; i < Z.n_rows; i++)
  {
    m(i, 0).zeros();
    int n1 = S_obs(i, 0).n_rows;
    int n2 = S_star(i, 0).n_rows;

    getCov(Z.row(i), Phi, Rho, Cov);
    mp_inv(i, 0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Cov *
      S_obs(i, 0).t();
    mp_inv(i, 0).submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Cov *
      S_star(i, 0).t();
    mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i, 0).submat(0, n1, n1 - 1,
                  n1 + n2 - 1).t();
    mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Cov *
      S_star(i, 0).t();

    // Get full precision matrix of observed and unobserved time points
    arma::pinv(mp_inv(i, 0), mp_inv(i, 0), 1e-20 * arma::datum::eps);

    mean_ph_obs(i, 0).zeros();
    mean_ph_star(i, 0).zeros();
    for(int j = 0; j < nu.n_cols; j ++){
      mean_ph_obs(i, 0) = mean_ph_obs(i, 0) + Z(i,j) * S_obs(i, 0) * nu.col(j);
      mean_ph_star(i, 0) = mean_ph_star(i, 0) + Z(i,j) * S_star(i, 0) *
        nu.col(j);
    }
    // Get variance
    arma::pinv(M(i, 0), mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1), 1e-20 * arma::datum::eps);
    // Compute mean
    m(i, 0) = mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs(i, 0)
                                                                 - f_obs(i, 0)) +
       mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star(i, 0);
  }
}

//' Computes m and M for all observations under common variance model
//'
//' @name compute_M_m
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Matrix of current Phi paramaters
//' @param nu Matrix containing the mean vectors as columns
//' @param mp_inv Matrix acting as a placeholder for mp inverse
//' @param mean_ph_obs vector acting as placeholder for mean of f_i at observed time points
//' @param mean_ph_star vector acting as placeholder for vector with length of unobserved time points
//' @param M Matrix acting as a placeholder for M

void compute_M_m(const arma::field<arma::mat>& S_obs,
                 const arma::field<arma::mat>& S_star,
                 const arma::field<arma::vec>& f_obs,
                 const arma::mat& Z,
                 const arma::mat& Phi,
                 const arma::mat& nu,
                 arma::field<arma::mat>& mp_inv,
                 arma::field<arma::vec>& mean_ph_obs,
                 arma::field<arma::vec>& mean_ph_star,
                 arma::field<arma::vec>& m,
                 arma::field<arma::mat>& M)
{
  for(int i =0; i < Z.n_rows; i++)
  {
    m(i, 0).zeros();
    int n1 = S_obs(i, 0).n_rows;
    int n2 = S_star(i, 0).n_rows;

    mp_inv(i, 0).submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Phi *
      Phi.t() * S_obs(i, 0).t();
    mp_inv(i, 0).submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Phi *
      Phi.t() * S_star(i, 0).t();
    mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv(i, 0).submat(0, n1,
           n1 - 1, n1 + n2 - 1).t();
    mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) *
      Phi * Phi.t() * S_star(i, 0).t();

    // Get full precision matrix of observed and unobserved time points
    arma::pinv(mp_inv(i, 0), mp_inv(i, 0), 1e-20 * arma::datum::eps);

    mean_ph_obs(i, 0).zeros();
    mean_ph_star(i, 0).zeros();
    for(int j = 0; j < nu.n_cols; j ++){
      mean_ph_obs(i, 0) = mean_ph_obs(i, 0) + Z(i,j) * S_obs(i, 0) * nu.col(j);
      mean_ph_star(i, 0) = mean_ph_star(i, 0) + Z(i,j) * S_star(i, 0) *
        nu.col(j);
    }
    // Get variance
    arma::pinv(M(i, 0), mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1), 1e-20 * arma::datum::eps);
    // Compute mean
    m(i, 0) = mp_inv(i, 0).submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs(i, 0)
                                                               - f_obs(i, 0)) +
      mp_inv(i, 0).submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star(i, 0);
  }
}

//' Computes m_i and M_i
//'
//' @name compute_mi_Mi
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Cube of current Phi paramaters
//' @param Rho Matrix with each row containing the elements of the upper triangular matrix
//' @param nu Matrix that contains mean vectors as columns
//' @param i Int indicating which M we are calculating
//' @param Cov Matrix containing placeholder for covariance matrix
//' @param mp_inv Matrix acting as a placeholder for mp inverse
//' @param mean_ph_obs vector acting as placeholder for mean of f_i at observed time points
//' @param mean_ph_star vector acting as placeholder for vector with length of unobserved time points
//' @param m Vector acting as a placeholder for m
//' @param M Matrix acting as a placeholder for M

void compute_mi_Mi(const arma::field<arma::mat>& S_obs,
                   const arma::field<arma::mat>& S_star,
                   const arma::field<arma::vec>& f_obs,
                   const arma::mat& Z,
                   const arma::cube& Phi,
                   const arma::mat& Rho,
                   const arma::mat& nu,
                   const int i,
                   arma::mat& Cov,
                   arma::mat& mp_inv,
                   arma::vec& mean_ph_obs,
                   arma::vec& mean_ph_star,
                   arma::vec& m,
                   arma::mat& M)
{
  m.zeros();
  int n1 = S_obs(i, 0).n_rows;
  int n2 = S_star(i, 0).n_rows;

  getCov(Z.row(i), Phi, Rho, Cov);
  mp_inv.submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Cov *
    S_obs(i, 0).t();
  mp_inv.submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Cov *
    S_star(i, 0).t();
  mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv.submat(0, n1, n1 - 1,
                n1 + n2 - 1).t();
  mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Cov *
    S_star(i, 0).t();

  // Get full precision matrix of observed and unobserved time points
  arma::pinv(mp_inv, mp_inv, 1e-20 * arma::datum::eps);

  mean_ph_obs.zeros();
  mean_ph_star.zeros();
  for(int j = 0; j < nu.n_cols; j ++){
    mean_ph_obs = mean_ph_obs + Z(i,j) * S_obs(i, 0) * nu.col(j);
    mean_ph_star = mean_ph_star + Z(i,j) * S_star(i, 0) * nu.col(j);
  }
  // Compute Covariance
  arma::pinv(M, mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1), 1e-20 * arma::datum::eps);
  // Compute mean
  m = mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs - f_obs(i, 0)) +
    mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star;
}

//' Computes m_i under common covariance matrix
//'
//' @name compute_mi_Mi
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param Z Matrix of current Z parameter
//' @param Phi Matrix of current Phi paramaters
//' @param i Int indicating which M we are calculating
//' @param mp_inv Matrix acting as a placeholder for mp inverse
//' @param mean_ph_obs vector acting as placeholder for mean of f_i at observed time points
//' @param mean_ph_star vector acting as placeholder for vector with length of unobserved time points
//' @param m Vector acting as a placeholder for m
//' @param M Matrix acting as a placeholder for M

void compute_mi_Mi(const arma::field<arma::mat>& S_obs,
                   const arma::field<arma::mat>& S_star,
                   const arma::field<arma::vec>& f_obs,
                   const arma::mat& Z,
                   const arma::mat& Phi,
                   const arma::mat& nu,
                   const int i,
                   arma::mat& mp_inv,
                   arma::vec& mean_ph_obs,
                   arma::vec& mean_ph_star,
                   arma::vec& m,
                   arma::mat& M)
{
  m.zeros();
  int n1 = S_obs(i, 0).n_rows;
  int n2 = S_star(i, 0).n_rows;

  mp_inv.submat(0, 0, n1 - 1, n1 - 1) = S_obs(i, 0) * Phi *
    Phi.t() * S_obs(i, 0).t();
  mp_inv.submat(0, n1, n1 - 1, n1 + n2 - 1) =  S_obs(i, 0) * Phi *
    Phi.t() * S_star(i, 0).t();
  mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) = mp_inv.submat(0, n1, n1 - 1,
                n1 + n2 - 1).t();
  mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) = S_star(i, 0) * Phi *
    Phi.t() * S_star(i, 0).t();

  // Get full precision matrix of observed and unobserved time points
  arma::pinv(mp_inv, mp_inv, 1e-20 * arma::datum::eps);

  mean_ph_obs.zeros();
  mean_ph_star.zeros();
  for(int j = 0; j < nu.n_cols; j ++){
    mean_ph_obs = mean_ph_obs + Z(i,j) * S_obs(i, 0) * nu.col(j);
    mean_ph_star = mean_ph_star + Z(i,j) * S_star(i, 0) * nu.col(j);
  }
  // Get variance
  arma::pinv(M, mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1), 1e-20 * arma::datum::eps);
  // Compute mean
  m = mp_inv.submat(n1, 0, n1 + n2 - 1, n1 - 1) * (mean_ph_obs - f_obs(i, 0)) +
    mp_inv.submat(n1, n1, n1 + n2 - 1, n1 + n2 - 1) * mean_ph_star;
}
