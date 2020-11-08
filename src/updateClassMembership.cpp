#include <RcppArmadillo.h>
#include <cmath>
#include "computeMM.h"

// [[Rcpp::depends(RcppArmadillo)]]

//' Updates the parameters pi_1 through pi_K
//'
//' @param Z Binary matrix of containing inclusion/exclusion to the various groups
//' @param alpha Hyperparameter for distribution of pi_i
//' @param K Number of clusters
//' @param iter Iteration of MCMC step
//' @param pi Matrix conatining all past, current, and future MCMC draws
//' @export
// [[Rcpp::export]]

void updatePi(const arma::mat& Z, const double alpha, const int K,
              const int iter, arma::mat& pi)
{
  for(int i = 0; i < K; i++)
  {
    pi(iter, i) = R::rbeta((alpha/K) + arma::accu(Z.col(i)),
       Z.n_rows - arma::accu(Z.col(i)) + 1);
  }
}

//' Gets generalized log determinant (product of positive eigen values)
//'
//' @param M Matrix that we want the determinant of
//' @return g_ldet Double countaining the generalized determinant
//' @export
// [[Rcpp::export]]

double g_ldet(const arma::mat& M)
{
  int N = M.n_rows;
  // initialize vector for eigenvalues
  arma::vec E = arma::zeros(N);
  // find eigen values
  eig_sym(E, M);
  double g_ldet =0;
  for(int i = 0; i < N; i++)
  {
    if(E(i) > 0)
    {
      g_ldet = g_ldet + log(E(i));
    }
  }
  return g_ldet;
}

//' Gets log-pdf of z given zeta
//'
//' @param M Cube that contains the M_i variance matrices
//' @param m Matrix that contains the m_i mean vectors
//' @param tilde_M Cube that contains the tilde_M_i variance matrices
//' @param tilde_m Matrix that contains the tilde_M_i mean vectors
//' @param f_obs Vector containing f at observed time points
//' @param f_star Vector containing f at unobserved time points
//' @param pi Vector containing the sampled pi for this iteration
//' @return lpdf_z Double contianing the log-pdf
//' @export
// [[Rcpp::export]]

double lpdf_z(const arma::mat& M, const arma::vec& m, const arma::mat& tilde_M,
              const arma::colvec& tilde_m, const arma::colvec& f_obs,
              const arma::colvec& f_star, const double pi_l,const double z_il,
              arma::mat pinv_M, arma::mat pinv_tilde_M)
{
  arma::pinv(pinv_M, M);
  arma::pinv(pinv_tilde_M, tilde_M);
  double g_ldet_M = g_ldet(M);
  double g_ldet_tilde_M = g_ldet(tilde_M);
  double lpdf_z = (-0.5 * g_ldet_M) - (0.5 * arma::dot(arma::pinv(M) *(f_obs -
                   M*m), (f_obs - M*m))) - (0.5 * g_ldet_tilde_M) - (0.5 *
                   arma::dot(arma::pinv(tilde_M) * (f_star -tilde_M *tilde_m),
                   (f_star -tilde_M *tilde_m))) + z_il * log(pi_l) +
                   (1 - z_il) * log(1 - pi_l);
  return lpdf_z;
}

//' Updates the ith row of the Z Matrix
//'
//' @param M Cube that contains all M matrices
//' @param M_ph Matrix that acts as a placeholder for the new M matrix
//' @param m Matrix that contains all m mean vectors
//' @param m_ph vector that acts as a placeholder for new m vector
//' @param tilde_M Matrix that contains the tilde_M_i variance matrix
//' @param tilde_m Vector that contains the tilde_M_i mean vector
//' @param f_obs Field of vectors containing f at observed time points
//' @param f_star Field of vectors containing f at unobserved time points
//' @param pi Vector containing the sampled pi for this iteration
//' @param iter Iteration of MCMC step
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param phi Cube of current phi paramaters
//' @param Z Cube that contains all past, current, and future MCMC draws
//' @export
// [[Rcpp::export]]

void updateZ_i(arma::cube& M, arma::mat M_ph, arma::mat& m, arma::vec& m_ph,
               arma::cube& tilde_M, const arma::mat& tilde_m, const arma::mat& f_obs,
               const arma::mat& f_star, const arma::vec pi, const int iter,
               const arma::field<arma::mat>& S_obs, const arma::cube phi,
               const arma::mat& nu, arma::cube& Z)
{
  computeM(S_obs, Z.slice(iter), phi, M);
  compute_m(S_obs, Z.slice(iter), phi, nu, m);

  for(int i = 0; i < Z.slice(iter).n_rows; i++)
  {

    for(int l = 0; l < Z.slice(iter).n_cols; l++)
    {
      computeMi(S_obs, Z.slice(iter), phi, i, M_ph);
      compute_mi(S_obs, Z.slice(iter), phi, nu, i, m_ph);


    }
  }

}


