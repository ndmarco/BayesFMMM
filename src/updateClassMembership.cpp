#include <RcppArmadillo.h>
#include <cmath>
#include "computeMM.H"
#include "CalculateCov.H"

// [[Rcpp::depends(RcppArmadillo)]]

//' Gets generalized log determinant (product of positive eigen values)
//'
//' @name g_ldet
//' @param S Vector of singular values
//' @return g_ldet Double countaining the generalized determinant
// [[Rcpp::export]]
double g_ldet(const arma::vec& S)
{

  double g_ldet = 0;
  for(int i = 0; i < S.n_elem; i++)
  {
    if(S(i) > 0){
      g_ldet = g_ldet + log(S(i));
    }
  }
  return g_ldet;
}

//' Gets log-pdf of z given zeta
//'
//' @name lpdf_z
//' @param f_obs Vector containing f at observed time points
//' @param f_star Vector containing f at unobserved time points
//' @param S_obs Matrix containing basis functions evaluated at observed time points
//' @param Phi Matrix containing covariance matrix
//' @param nu Matrix containing mean vectors as the columns
//' @param pi_l double containing the lth element of pi
//' @param Z Matrix containing the elements of Z
//' @param i int containing row of Z we are finding pdf of
//' @param mean_UV Field of matrices containing placeholder for U and V matrices of SVD of covariance matrix
//' @param mean_S Field of vectors containg placeholder for S (diag matrix) of SVD of covariance matrix
//' @param mean_ph_obs vector containing placeholder for mean of observed data
//' @return lpdf_z double contianing the log-pdf

double lpdf_z(const arma::vec& f_obs,
              const arma::mat& S_obs,
              const arma::mat& Cov,
              const arma::mat& nu,
              const arma::vec& pi,
              const arma::mat& Z,
              const int rank,
              const int i,
              arma::field<arma::mat>& mean_UV,
              arma::field<arma::vec>& mean_S,
              arma::vec& mean_ph_obs)
{
  arma::svd(mean_UV(i,0), mean_S(i,0), mean_UV(i,1), S_obs * Cov * S_obs.t());
  for(int j = 0; j < rank; j++)
  {
    if(mean_S(i,0)(j) <= 0)
    {
      mean_S(i,0)(j) = 0;
      mean_S(i,1)(j) = 0;

    }else
    {
      mean_S(i,1)(j) = 1 /  mean_S(i,0)(j);
    }
  }
  double g_ldet1 = g_ldet(mean_S(i,0));
  mean_ph_obs.zeros();
  for(int l = 0; l < nu.n_cols; l++){
    mean_ph_obs = mean_ph_obs + Z(i,l) * S_obs * nu.col(l);
  }

  double lpdf_z = (-0.5 * g_ldet1) - (0.5 * arma::dot(mean_UV(i,1) *
                   arma::diagmat(mean_S(i,1)) * mean_UV(i,0).t() *
                   (f_obs - mean_ph_obs), f_obs - mean_ph_obs));
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
//' @param pi Vector containing the sampled pi for this iteration
//' @param iter Iteration of MCMC step
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Phi Cube of current Phi paramaters
//' @param Rho Matrix with each row containing the elements of the upper triangular matrix
//' @param nu Matrix that contains all current nu paramaters
//' @param Cov Matrix containing placeholder for covariance matrix
//' @param m Field of Vectors that contains all m mean vectors
//' @param mean_ph_obs Field of vectors that serve as a placeholder of computations
//' @param Z_ph Matrix that acts as a placeholder for Z
//' @param Z Cube that contains all past, current, and future MCMC draws

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::cube& Phi,
             const arma::mat& Rho,
             const double& sigma_phi,
             const arma::mat& nu,
             const int rank,
             arma::field<arma::mat>& mean_UV,
             arma::field<arma::vec>& mean_S,
             arma::mat& Cov,
             arma::field<arma::vec>& mean_ph_obs,
             arma::mat& Z_ph,
             arma::cube& Z)
{
  double z_lpdf = 0;
  double z_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;

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
      getCov(Z.slice(iter).row(i), Phi, Rho, sigma_phi, Cov);
      z_lpdf = lpdf_z(f_obs(i,0), S_obs(i,0), Cov, nu, pi,
                      Z.slice(iter), rank, i, mean_UV, mean_S,
                      mean_ph_obs(i,0));
      getCov(Z_ph.row(i), Phi, Rho, sigma_phi, Cov);
      z_new_lpdf = lpdf_z(f_obs(i,0), S_obs(i,0), Cov, nu, pi,
                          Z_ph, rank, i, mean_UV, mean_S, mean_ph_obs(i,0));
      acceptance_prob = z_new_lpdf - z_lpdf;
      rand_unif_var = R::runif(0,1);

      if(log(rand_unif_var) < acceptance_prob)
      {
        // Accept new state and update parameters
        Z.slice(iter).row(i) = Z_ph.row(i);
      } else
      {
        Z_ph.row(i) = Z.slice(iter).row(i);
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
//' @param pi Vector containing the sampled pi for this iteration
//' @param iter Iteration of MCMC step
//' @param S_obs Field of Matrices containing basis functions evaluated at observed time points
//' @param Phi Matrix of current Phi paramaters
//' @param nu Matrix that contains all current nu paramaters
//' @param Z_ph Matrix that acts as a placeholder for the new Z matrix
//' @param mean_ph_obs Field of vectors that serve as a placeholder of computations
//' @param Z Cube that contains all past, current, and future MCMC draws

void updateZ(const arma::field<arma::vec>& f_obs,
             const arma::vec& pi,
             const int iter,
             const arma::field<arma::mat>& S_obs,
             const arma::mat& Phi,
             const double& sigma_phi,
             const arma::mat& nu,
             const int rank,
             arma::field<arma::mat>& mean_UV,
             arma::field<arma::vec>& mean_S,
             arma::mat& Cov,
             arma::field<arma::vec>& mean_ph_obs,
             arma::mat& Z_ph,
             arma::cube& Z)
{
  double z_lpdf = 0;
  double z_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;

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
      Cov = Phi * Phi.t();
      Cov.diag() += sigma_phi;
      z_lpdf = lpdf_z(f_obs(i,0), S_obs(i,0), Cov, nu, pi, Z.slice(iter), rank, i,
                      mean_UV, mean_S, mean_ph_obs(i,0));
      z_new_lpdf = lpdf_z(f_obs(i,0), S_obs(i,0), Cov, nu, pi, Z_ph, rank, i,
                          mean_UV, mean_S, mean_ph_obs(i,0));
      acceptance_prob = z_new_lpdf - z_lpdf;
      rand_unif_var = R::runif(0,1);

      if(log(rand_unif_var) < acceptance_prob)
      {
        // Accept new state and update parameters
        Z.slice(iter).row(i) = Z_ph.row(i);
      } else
      {
        Z_ph.row(i) = Z.slice(iter).row(i);
      }
    }
  }
  // Update next iteration
  if(iter < (Z.n_slices - 1))
  {
    Z.slice(iter + 1) = Z.slice(iter);
  }
}




