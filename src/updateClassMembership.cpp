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

void updatePi(const arma::mat& Z,
              const double alpha,
              const int K,
              const int iter,
              arma::mat& pi)
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
  // make sure M is symmetric
  arma::eig_sym(E, ((M + M.t()) /2));
  double g_ldet = 0;
  for(int i = 0; i < N; i++)
  {
    if(E(i) > 1e-20)
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

double lpdf_z(const arma::mat& M,
              const arma::vec& m,
              const arma::mat& tilde_M,
              const arma::vec& tilde_m,
              const arma::vec& f_obs,
              const arma::vec& f_star,
              const double pi_l,
              const double z_il,
              arma::mat pinv_M,
              arma::mat pinv_tilde_M)
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

//' Updates the Z Matrix
//'
//' @param f_obs Field of vectors containing f at observed time points
//' @param f_star Field of vectors containing f at unobserved time points
//' @param pi Vector containing the sampled pi for this iteration
//' @param iter Iteration of MCMC step
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param phi Cube of current phi paramaters
//' @param nu Matrix that contains all current nu paramaters
//' @param M Field of Matrices that contains all M matrices
//' @param M_ph Field of Matrices that acts as a placeholder for the new M matrix
//' @param pinv_M Field of Matrices that acts as a placeholder for g-inverse of M
//' @param Z_ph Matrix that acts as a placeholder for the new Z matrix
//' @param m Field of Vectors that contains all m mean vectors
//' @param m_ph Field of Vectors that acts as a placeholder for new m vector
//' @param z_ph Matrix that acts as a placeholder for Z
//' @param tilde_M Field of Matrices that contains the tilde_M_i variance matrix
//' @param tilde_M_ph Field of Matrices that acts as a placeholder for tilde_M
//' @param pinv_tilde_M Field of Matrices that acts as a placeholder for g-inverse of tilde M
//' @param tilde_m Vector that contains the tilde_M_i mean vector
//' @param Z_plus Field of Matrices acting as a placeholder for Z_plus
//' @param A_plus Matrix acting as a placeholder for A_plus
//' @param C Matrix acting as a placeholder for C
//' @param Z Cube that contains all past, current, and future MCMC draws

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::field<arma::vec>& f_star,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::field<arma::mat>& S_star,
             const arma::cube& phi,
             const arma::mat& nu,
             arma::field<arma::mat>& M,
             arma::field<arma::mat>& M_ph,
             arma::field<arma::mat>& pinv_M,
             arma::field<arma::vec>& m,
             arma::field<arma::vec>& m_ph,
             arma::mat& Z_ph,
             arma::field<arma::mat>& tilde_M,
             arma::field<arma::mat>& tilde_M_ph,
             arma::field<arma::mat>& pinv_tilde_M,
             arma::field<arma::vec>& tilde_m,
             arma::field<arma::vec>& tilde_m_ph,
             arma::field<arma::mat>& mp_inv,
             arma::cube& Z)
{
  double z_lpdf = 0;
  double z_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;
  computeM(S_obs, Z.slice(iter), phi, M);
  compute_m(S_obs, Z.slice(iter), phi, nu, m);
  compute_tildeM_tildem(S_star, S_obs, f_obs, Z.slice(iter), phi, nu, mp_inv,
                        tilde_M, tilde_m);
  tilde_m_ph = tilde_m;
  tilde_M_ph = tilde_M;
  M_ph = M;
  m_ph = m;
  Z_ph = Z.slice(iter);
  for(int i = 0; i < Z.slice(iter).n_rows; i++)
  {
    for(int l = 0; l < Z.slice(iter).n_cols; l++)
    {
      // Propose new state
      Z_ph(i,l) = R::rbinom(1, 0.5);
      // Compute lpdf to see if we accept or reject new state
      if(Z_ph(i,l) != Z.slice(iter)(i,l) && arma::accu(Z_ph.row(i)) > 0)
      {
        computeMi(S_obs, Z_ph, phi, i, M_ph(i,0));
        compute_mi(S_obs, Z_ph, phi, nu, i, m_ph(i,0));
        compute_tildeMi_tildemi(S_star, S_obs, f_obs, Z.slice(iter), phi, nu, i,
                                mp_inv, tilde_M, tilde_m);
        z_lpdf = lpdf_z(M(i,0), m(i,0), tilde_M(i,0), tilde_m(i,0), f_obs(i,0),
                        f_star(i,0), pi(l), Z.slice(iter)(i,l), pinv_M(i,0),
                        pinv_tilde_M(i,0));
        z_new_lpdf = lpdf_z(M_ph(i,0), m_ph(i,0), tilde_M_ph(i,0), tilde_m_ph(i,0),
                            f_obs(i,0), f_star(i,0), pi(l), Z_ph(i,l), pinv_M(i,0),
                            pinv_tilde_M(i,0));
        acceptance_prob = z_new_lpdf - z_lpdf;
        rand_unif_var = R::runif(0,1);

        if(log(rand_unif_var) < acceptance_prob)
        {
          // Accept new state and update parameters
          Z.slice(iter)(i,l) = Z_ph(i,l);
          M(i,0) = M_ph(i,0);
          m(i,0) = m_ph(i,0);
          tilde_M(i,0) = tilde_M_ph(i,0);
          tilde_m(i,0) = tilde_m_ph(i,0);
        } else
        {
          Z_ph(i,l) = Z.slice(iter)(i,l);
          tilde_m_ph(i,0) = tilde_m(i,0);
          tilde_M_ph(i,0) = tilde_M(i,0);
          M_ph(i,0) = M(i,0);
          m_ph(i,0) = m(i,0);
        }
      }
    }
  }
  // Update next iteration
  if(iter < (Z.n_slices - 1))
  {
    Z.slice(iter + 1) = Z.slice(iter);
  }
}


//' Updates the Z Matrix for single covariance matrix
//'
//' @param f_obs Field of vectors containing f at observed time points
//' @param f_star Field of vectors containing f at unobserved time points
//' @param pi Vector containing the sampled pi for this iteration
//' @param iter Iteration of MCMC step
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param phi Matrix of current phi paramaters
//' @param nu Matrix that contains all current nu paramaters
//' @param M Field of Matrices that contains all M matrices
//' @param M_ph Field of Matrices that acts as a placeholder for the new M matrix
//' @param pinv_M Field of Matrices that acts as a placeholder for g-inverse of M
//' @param Z_ph Matrix that acts as a placeholder for the new Z matrix
//' @param m Field of Vectors that contains all m mean vectors
//' @param m_ph Field of Vectors that acts as a placeholder for new m vector
//' @param z_ph Matrix that acts as a placeholder for Z
//' @param tilde_M Field of Matrices that contains the tilde_M_i variance matrix
//' @param tilde_M_ph Field of Matrices that acts as a placeholder for tilde_M
//' @param pinv_tilde_M Field of Matrices that acts as a placeholder for g-inverse of tilde M
//' @param tilde_m Vector that contains the tilde_M_i mean vector
//' @param Z_plus Field of Matrices acting as a placeholder for Z_plus
//' @param A_plus Matrix acting as a placeholder for A_plus
//' @param C Matrix acting as a placeholder for C
//' @param Z Cube that contains all past, current, and future MCMC draws


void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::field<arma::vec>& f_star,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::field<arma::mat>& S_star,
             const arma::mat& phi,
             const arma::mat& nu,
             arma::field<arma::mat>& M,
             arma::field<arma::mat>& M_ph,
             arma::field<arma::mat>& pinv_M,
             arma::field<arma::vec>& m,
             arma::field<arma::vec>& m_ph,
             arma::mat& Z_ph,
             arma::field<arma::mat>& tilde_M,
             arma::field<arma::mat>& tilde_M_ph,
             arma::field<arma::mat>& pinv_tilde_M,
             arma::field<arma::vec>& tilde_m,
             arma::field<arma::vec>& tilde_m_ph,
             arma::field<arma::mat>& mp_inv,
             arma::cube& Z)
{
  double z_lpdf = 0;
  double z_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;
  computeM(S_obs, Z.slice(iter), phi, M);
  compute_m(S_obs, Z.slice(iter), phi, nu, m);
  compute_tildeM_tildem(S_star, S_obs, f_obs, Z.slice(iter), phi, nu, mp_inv,
                        tilde_M, tilde_m);
  tilde_m_ph = tilde_m;
  tilde_M_ph = tilde_M;
  M_ph = M;
  m_ph = m;
  Z_ph = Z.slice(iter);
  for(int i = 0; i < Z.slice(iter).n_rows; i++)
  {
    for(int l = 0; l < Z.slice(iter).n_cols; l++)
    {
      // Propose new state
      Z_ph(i,l) = R::rbinom(1, 0.5);
      // Compute lpdf to see if we accept or reject new state
      if(Z_ph(i,l) != Z.slice(iter)(i,l) && arma::accu(Z_ph.row(i)) > 0)
      {
        computeMi(S_obs, Z_ph, phi, i, M_ph(i,0));
        compute_mi(S_obs, Z_ph, phi, nu, i, m_ph(i,0));
        compute_tildeMi_tildemi(S_star, S_obs, f_obs, Z.slice(iter), phi, nu, i,
                                mp_inv, tilde_M, tilde_m);
        z_lpdf = lpdf_z(M(i,0), m(i,0), tilde_M(i,0), tilde_m(i,0), f_obs(i,0),
                        f_star(i,0), pi(l), Z.slice(iter)(i,l), pinv_M(i,0),
                        pinv_tilde_M(i,0));
        z_new_lpdf = lpdf_z(M_ph(i,0), m_ph(i,0), tilde_M_ph(i,0), tilde_m_ph(i,0),
                            f_obs(i,0), f_star(i,0), pi(l), Z_ph(i,l), pinv_M(i,0),
                            pinv_tilde_M(i,0));
        acceptance_prob = z_new_lpdf - z_lpdf;
        rand_unif_var = R::runif(0,1);

        if(log(rand_unif_var) < acceptance_prob)
        {
          // Accept new state and update parameters
          Z.slice(iter)(i,l) = Z_ph(i,l);
          M(i,0) = M_ph(i,0);
          m(i,0) = m_ph(i,0);
          tilde_M(i,0) = tilde_M_ph(i,0);
          tilde_m(i,0) = tilde_m_ph(i,0);
        } else
        {
          Z_ph(i,l) = Z.slice(iter)(i,l);
          tilde_m_ph(i,0) = tilde_m(i,0);
          tilde_M_ph(i,0) = tilde_M(i,0);
          M_ph(i,0) = M(i,0);
          m_ph(i,0) = m(i,0);
        }
      }
    }
  }
  // Update next iteration
  if(iter < (Z.n_slices - 1))
  {
    Z.slice(iter + 1) = Z.slice(iter);
  }
}


