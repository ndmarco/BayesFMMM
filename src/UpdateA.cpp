#include <RcppArmadillo.h>
#include <cmath>
#include <RcppDist.h>

double lpdf_a1(const double& alpha_1l,
               const double& beta_1l,
               const double& a,
               const double& delta)
{
  double lpdf = -lgamma(a) + (a - 1) * log(delta) + (alpha_1l - 1) * log(a) -
    (a * beta_1l);
  return lpdf;
}

double lpdf_a2(const double& alpha_2l,
               const double& beta_2l,
               const double& a,
               const arma::vec& delta)
{
  double lpdf = -(delta.n_elem - 1) * lgamma(a) + (alpha_2l - 1) * log(a) -
    (a * beta_2l);
  for(int i = 1; i < delta.n_elem; i ++)
  {
    lpdf = lpdf + (a - 1) * log(delta(i));
  }
  return lpdf;
}

// void UpdateA(const arma::vec& alpha_1l,
//              const arma::vec& beta_1l,
//              const arma::vec& alpha_2l,
//              const arma::vec& beta_2l,
//              const arma::mat& delta,
//              const double var_epsilon1,
//              const double var_epsilon2,
//              const int iter,
//              arma::cube a)
// {
//   double a_new_lpdf = 0;
//   double acceptance_prob = 0;
//   double rand_unif_var = 0;
//   double new_a = 0;
//
//   // caluclate first lpdf
//   double a_lpdf = lpdf_a1(alpha_1l(0), beta_1l(0), a(0, 0, iter), delta(0,0));
//   for(int i = 0; i < a.n_rows; i++)
//   {
//     for(int j = 0; j < a.n_cols; j++)
//     {
//       if(i == 0)
//       {
//         new_a = r_truncnorm(a(i, j, iter), var_epsilon1 / beta_1l, 0,
//                         std::numeric_limits<double>::infinity());
//         a_new_lpdf = lpdf_a1(alpha_1l(j), beta_1l(j), a(i, j, iter), delta(i,j));
//
//         acceptance_prob;
//       }
//
//     }
//   }
// }
