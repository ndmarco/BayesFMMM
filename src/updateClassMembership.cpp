#include <RcppArmadillo.h>
#include <cmath>
#include "computeMM.h"

// [[Rcpp::depends(RcppArmadillo)]]

//' Gets generalized log determinant (product of positive eigen values)
//'
//' @name g_ldet
//' @param M Matrix that we want the determinant of
//' @return g_ldet Double countaining the generalized determinant
// [[Rcpp::export]]
double g_ldet(const arma::mat& M, const int rank)
{
  // initialize vector for eigenvalues
  arma::vec E = arma::zeros(M.n_rows);
  // find eigen values
  // make sure M is symmetric
  arma::eig_sym(E, ((M + M.t()) /2));
  double g_ldet = 0;
  for(int i = E.n_elem - 1; i >= E.n_elem - rank; i--)
  {
    if(E(i) > 0){
      g_ldet = g_ldet + log(E(i));
    }
  }
  return g_ldet;
}

//' Gets log-pdf of z given zeta
//'
//' @name lpdf_z
//' @param M Cube that contains the M_i variance matrices
//' @param m Matrix that contains the m_i mean vectors
//' @param f_obs Vector containing f at observed time points
//' @param f_star Vector containing f at unobserved time points
//' @param S_obs Matrix containing basis functions evaluated at observed time points
//' @param phi Matrix containing covariance matrix
//' @param nu Matrix containing mean vectors as the columns
//' @param pi_l double containing the lth element of pi
//' @param Z Matrix containing the elements of Z
//' @param i int containing row of Z we are finding pdf of
//' @param j int containing column of Z we are finding the pdf of
//' @param mean_ph_obs vector containing placeholder for mean of observed data
//' @return lpdf_z double contianing the log-pdf

double lpdf_z(const arma::mat& M,
              const arma::vec& m,
              const arma::vec& f_obs,
              const arma::vec& f_star,
              const arma::mat& S_obs,
              const arma::mat& phi,
              const arma::mat& nu,
              const arma::vec& pi,
              const arma::mat& Z,
              const int i,
              arma::vec& mean_ph_obs)
{
  double g_ldet1 = g_ldet(S_obs * phi * phi.t() * S_obs.t(), std::min(phi.n_cols,
                                              std::min(phi.n_rows, S_obs.n_rows)));
  double g_ldetM = g_ldet(M, std::min(phi.n_cols, std::min(phi.n_rows, M.n_rows)));
  mean_ph_obs.zeros();
  for(int l = 0; l < nu.n_cols; l++){
    mean_ph_obs = mean_ph_obs + Z(i,l) * S_obs * nu.col(l);
  }
  double lpdf_z = (-0.5 * g_ldet1) - (0.5 * arma::dot((f_obs - mean_ph_obs).t() *
                   arma::pinv(S_obs * phi * phi.t() * S_obs.t()), f_obs - mean_ph_obs));
  lpdf_z = lpdf_z - (0.5 * g_ldetM) - (0.5 * arma::dot((f_star - M * m).t() *
    arma::pinv(M),(f_star - M * m)));
  for(int j = 0; j < Z.n_cols; j++)
  {
    lpdf_z = lpdf_z + (Z(i,j) * log(pi(j))) + (1- Z(i,j)) * log(1 - pi(j));
  }

  return lpdf_z;
}

//' Updates the Z Matrix
//'
//' @name UpdateZ
//' @param f_obs Field of vectors containing f at observed time points
//' @param f_star Field of vectors containing f at unobserved time points
//' @param pi Vector containing the sampled pi for this iteration
//' @param iter Iteration of MCMC step
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param S_star Field of Matrices containing basis functions evaluated at unobserved time points
//' @param phi Cube of current phi paramaters
//' @param nu Matrix that contains all current nu paramaters
//' @param Map a map that contains the index of covariance matrices
//' @param M Field of Matrices that contains all M matrices
//' @param M_ph Field of Matrices that acts as a placeholder for the new M matrix
//' @param Z_ph Matrix that acts as a placeholder for the new Z matrix
//' @param m Field of Vectors that contains all m mean vectors
//' @param m_ph Field of Vectors that acts as a placeholder for new m vector
//' @param mean_ph_obs Field of vectors that serve as a placeholder of computations
//' @param mean_ph_star Field of vectors that serve as a placeholder of computations
//' @param Z_ph Matrix that acts as a placeholder for Z
//' @param mp_inv Field of matrices that act as a placeholder for joint covariance matrix
//' @param Z Cube that contains all past, current, and future MCMC draws

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::field<arma::vec>& f_star,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::field<arma::mat>& S_star,
             const arma::cube& phi,
             const arma::mat& nu,
             const std::map<double, int> Map,
             arma::field<arma::mat>& M,
             arma::field<arma::mat>& M_ph,
             arma::field<arma::vec>& m,
             arma::field<arma::vec>& m_ph,
             arma::field<arma::vec>& mean_ph_obs,
             arma::field<arma::vec>& mean_ph_star,
             arma::mat& Z_ph,
             arma::field<arma::mat>& mp_inv,
             arma::cube& Z)
{
  double z_lpdf = 0;
  double z_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;
  compute_M_m(S_obs, S_star, f_obs, Z.slice(iter), phi, Map, nu, mp_inv, mean_ph_obs,
              mean_ph_star, m, M);
  m_ph = m;
  M_ph = M;
  Z_ph = Z.slice(iter);
  for(int i = 0; i < Z.slice(iter).n_rows; i++)
  {
    for(int l = 0; l < Z.slice(iter).n_cols; l++)
    {
      // Propose new state
      Z_ph(i,l) = R::rbinom(1, 0.5);
    }
    if(arma::accu(Z_ph.row(i)) > 0)
    {
      compute_mi_Mi(S_obs, S_star, f_obs, Z_ph, phi, Map, nu, i, mp_inv(i,0),
                    mean_ph_obs(i,0), mean_ph_star(i,0), m_ph(i,0), M_ph(i,0));
      z_lpdf = lpdf_z(M(i,0), m(i,0), f_obs(i,0), f_star(i,0), S_obs(i,0),
                      phi.slice(Map.at(get_ind(Z.slice(iter).row(i).t()))), nu, pi,
                      Z.slice(iter), i, mean_ph_obs(i,0));
      z_new_lpdf = lpdf_z(M_ph(i,0), m_ph(i,0), f_obs(i,0), f_star(i,0),
                          S_obs(i,0), phi.slice(Map.at(get_ind(Z_ph.row(i).t()))), nu,
                          pi, Z_ph, i, mean_ph_obs(i,0));
      acceptance_prob = z_new_lpdf - z_lpdf;
      rand_unif_var = R::runif(0,1);

      if(log(rand_unif_var) < acceptance_prob)
      {
        // Accept new state and update parameters
        Z.slice(iter).row(i) = Z_ph.row(i);
        M(i,0) = M_ph(i,0);
        m(i,0) = m_ph(i,0);
      } else
      {
        Z_ph.row(i) = Z.slice(iter).row(i);
        M_ph(i,0) = M(i,0);
        m_ph(i,0) = m(i,0);
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
//' @param mean_ph_obs Field of vectors that serve as a placeholder of computations
//' @param mean_ph_star Field of vectors that serve as a placeholder of computations
//' @param Z_ph Matrix that acts as a placeholder for Z
//' @param mp_inv Field of matrices that act as a placeholder for joint covariance matrix
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
             arma::field<arma::vec>& m,
             arma::field<arma::vec>& m_ph,
             arma::field<arma::vec>& mean_ph_obs,
             arma::field<arma::vec>& mean_ph_star,
             arma::mat& Z_ph,
             arma::field<arma::mat>& mp_inv,
             arma::cube& Z)
{
  double z_lpdf = 0;
  double z_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;
  compute_M_m(S_obs, S_star, f_obs, Z.slice(iter), phi, nu, mp_inv, mean_ph_obs,
              mean_ph_star, m, M);
  m_ph = m;
  M_ph = M;
  Z_ph = Z.slice(iter);
  for(int i = 0; i < Z.slice(iter).n_rows; i++)
  {
    for(int l = 0; l < Z.slice(iter).n_cols; l++)
    {
      // Propose new state
      Z_ph(i,l) = R::rbinom(1, 0.5);
    }
    if(arma::accu(Z_ph.row(i)) > 0)
    {
      compute_mi_Mi(S_obs, S_star, f_obs, Z_ph, phi, nu, i, mp_inv(i,0),
                    mean_ph_obs(i,0), mean_ph_star(i,0), m_ph(i,0), M_ph(i,0));
      z_lpdf = lpdf_z(M(i,0), m(i,0), f_obs(i,0), f_star(i,0), S_obs(i,0),
                      phi, nu, pi,
                      Z.slice(iter), i, mean_ph_obs(i,0));
      z_new_lpdf = lpdf_z(M_ph(i,0), m_ph(i,0), f_obs(i,0), f_star(i,0),
                          S_obs(i,0), phi, nu,
                          pi, Z_ph, i, mean_ph_obs(i,0));
      acceptance_prob = z_new_lpdf - z_lpdf;
      rand_unif_var = R::runif(0,1);

      if(log(rand_unif_var) < acceptance_prob)
      {
        // Accept new state and update parameters
        Z.slice(iter).row(i) = Z_ph.row(i);
        M(i,0) = M_ph(i,0);
        m(i,0) = m_ph(i,0);
      } else
      {
        Z_ph.row(i) = Z.slice(iter).row(i);
        M_ph(i,0) = M(i,0);
        m_ph(i,0) = m(i,0);
      }
    }
  }
  // Update next iteration
  if(iter < (Z.n_slices - 1))
  {
    Z.slice(iter + 1) = Z.slice(iter);
  }
}


