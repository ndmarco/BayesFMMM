#include <RcppArmadillo.h>
#include <cmath>

//' Calculates covariance matrix
//'
//' @name getCov
//' @param Z Vector that conatins the class memebership
//' @param Phi Cube that contains the K covariance matrices
//' @param Rho Matrix with each row containing the elements of the upper triangular matrix
//' @param Cov Matrix acting as placeholder for covariance matrix
//' @export
// [[Rcpp::export]]

void getCov(const arma::rowvec& Z,
            const arma::cube& Phi,
            const arma::mat& Rho,
            arma::mat& Cov)
{
  Cov.zeros();
  int k = Phi.n_slices;
  int counter = 0;
  int counter2 = 0;
  for(int i = 0; i < Phi.n_slices; i++)
  {
    if(Z(i) == 1)
    {
      Cov = Cov + Phi.slice(i) * Phi.slice(i).t();
      // if(i < Phi.n_slices - 1)
      // {
      //   for(int j = i + 1; j < Phi.n_slices; j ++)
      //   {
      //     if(Z(j) == 1)
      //     {
      //       counter2 = 0;
      //       for(int l = 0; l < Cov.n_cols; l++)
      //       {
      //         for(int m = l; m < Cov.n_rows; m++)
      //         {
      //           Cov(l, m) = Cov(l, m) + (Rho(counter2, counter) * arma::norm(Phi.slice(i).row(l), 2)
      //                                      * arma::norm(Phi.slice(j).row(m), 2));
      //           Cov(m, l) = Cov(l, m);
      //           counter2 = counter2 + 1;
      //         }
      //       }
      //     }
      //     counter = counter + 1;
      //   }
      // }
    }else
    {
      counter = counter + (k - i - 1);
    }
  }
}

//'Computes log pdf of phi
//'
//' @name lpdf_phi
//' @param M Cube that contains the M_i variance matrices
//' @param m Matrix that contains the m_i mean vectors
//' @param f_obs Vector containing f at observed time points
//' @param f_star Vector containing f at unobserved time points
//' @param S_obs Matrix containing basis functions evaluated at observed time points
//' @param phi Matrix containing covariance matrix
//' @param nu Matrix containing mean vectors as the columns
//' @param pi_l double containing the lth element of pi
//' @param Z Matrix containing the elements of Z
