#include <RcppArmadillo.h>
#include <cmath>
#include <truncnorm.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

//' computes the log pdf of a_1j
//'
//' @name lpdf_a1
//' @param alpha_1l Double containing hyperparameter for a
//' @param beta_1l Double containing hyperparameter for a
//' @param a Double containing value of a
//' @param delta double contianing value of delta
double lpdf_a1(const double& alpha_1l,
               const double& beta_1l,
               const double& a,
               const double& delta)
{
  double lpdf = -lgamma(a) + (a - 1) * log(delta) + (alpha_1l - 1) * log(a) -
    (a * beta_1l);
  return lpdf;
}

//' computes the log pdf of a_2j
//'
//' @name lpdf_a2
//' @param alpha_2l Double containing hyperparameter for a
//' @param beta_2l Double containing hyperparameter for a
//' @param a Double containing value of a
//' @param delta double contianing value of delta
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

//' updates the a parameters for individualized covariance matrix
//'
//' @name UpdateA
//' @param alpha_1l Vector containing hyperparameters for a
//' @param beta_1l Vector containing hyperparameters for a
//' @param alpha_2l Vector containing hyperparameters for a
//' @param beta_2l Vector containing hyperparameters for a
//' @param delta mat contianing value of delta
//' @param var_epsilon1 double containing hyperparameter epsilon1
//' @param var_epsilon2 double containing hyperparameter epsilon2
//' @param iter double containing MCMC iteration
//' @param a Cube containing values of a
void UpdateA(const arma::vec& alpha_1l,
             const arma::vec& beta_1l,
             const arma::vec& alpha_2l,
             const arma::vec& beta_2l,
             const arma::mat& delta,
             const double& var_epsilon1,
             const double& var_epsilon2,
             const int iter,
             arma::cube& a)
{
  double a_lpdf = 0;
  double a_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;
  double new_a = 0;

  // caluclate first lpdf
  for(int i = 0; i < a.n_rows; i++)
  {
    for(int j = 0; j < a.n_cols; j++)
    {
      if(i == 0)
      {
        a_lpdf = lpdf_a1(alpha_1l(j), beta_1l(j), a(i, j, iter), delta(i,j));
        new_a = r_truncnorm(a(i, j, iter), var_epsilon1 / beta_1l(j), 0,
                        std::numeric_limits<double>::infinity());

        a_new_lpdf = lpdf_a1(alpha_1l(j), beta_1l(j), new_a, delta(i,j));

        acceptance_prob = (a_new_lpdf + d_truncnorm(a(i,j, iter), new_a,
                                                   var_epsilon1 / beta_1l(j), 0,
                                                   std::numeric_limits<double>::infinity(),
                                                   1)) - a_lpdf -
                                        d_truncnorm(new_a, a(i,j, iter),
                                                   var_epsilon1 / beta_1l(j), 0,
                                                   std::numeric_limits<double>::infinity(),
                                                   1);
        rand_unif_var = R::runif(0,1);

        if(log(rand_unif_var) < acceptance_prob)
        {
          // Accept new state and update parameters
          a(i, j, iter) = new_a;
        }
      }else
      {
        a_lpdf = lpdf_a2(alpha_2l(j), beta_2l(j), a(i, j, iter), delta.col(j));
        new_a = r_truncnorm(a(i, j, iter), var_epsilon1 / beta_1l(j), 0,
                            std::numeric_limits<double>::infinity());

        a_new_lpdf = lpdf_a2(alpha_2l(j), beta_2l(j), new_a, delta.col(j));

        acceptance_prob = (a_new_lpdf + d_truncnorm(a(i,j, iter), new_a,
                                                    var_epsilon1 / beta_2l(j), 0,
                                                    std::numeric_limits<double>::infinity(),
                                                    1)) - a_lpdf -
                                                      d_truncnorm(new_a, a(i,j, iter),
                                                                  var_epsilon1 / beta_2l(j), 0,
                                                                  std::numeric_limits<double>::infinity(),
                                                                  1);
        rand_unif_var = R::runif(0,1);

        if(log(rand_unif_var) < acceptance_prob)
        {
          // Accept new state and update parameters
          a(i, j, iter) = new_a;
        }
      }

    }
  }
}

//' updates the a parameters for single covariance matrix
//'
//' @name UpdateA
//' @param alpha_1l double containing hyperparameters for a
//' @param beta_1l double containing hyperparameters for a
//' @param alpha_2l double containing hyperparameters for a
//' @param beta_2l double containing hyperparameters for a
//' @param delta vec contianing value of delta
//' @param var_epsilon1 double containing hyperparameter epsilon1
//' @param var_epsilon2 double containing hyperparameter epsilon2
//' @param iter int containing MCMC iteration
//' @param a mat containing values of a
void UpdateA(const double& alpha_1l,
             const double& beta_1l,
             const double& alpha_2l,
             const double& beta_2l,
             const arma::vec& delta,
             const double& var_epsilon1,
             const double& var_epsilon2,
             const int iter,
             arma::mat& a)
{
  double a_lpdf = 0;
  double a_new_lpdf = 0;
  double acceptance_prob = 0;
  double rand_unif_var = 0;
  double new_a = 0;

  // caluclate first lpdf
  for(int i = 0; i < a.n_rows; i++)
  {
    if(i == 0)
    {
      a_lpdf = lpdf_a1(alpha_1l, beta_1l, a(i, iter), delta(i));
      new_a = r_truncnorm(a(i, iter), var_epsilon1 / beta_1l, 0,
                          std::numeric_limits<double>::infinity());

      a_new_lpdf = lpdf_a1(alpha_1l, beta_1l, new_a, delta(i));

      acceptance_prob = (a_new_lpdf + d_truncnorm(a(i, iter), new_a,
                                                  var_epsilon1 / beta_1l, 0,
                                                  std::numeric_limits<double>::infinity(),
                                                  1)) - a_lpdf -
                                                    d_truncnorm(new_a, a(i, iter),
                                                                var_epsilon1 / beta_1l, 0,
                                                                std::numeric_limits<double>::infinity(),
                                                                1);
      rand_unif_var = R::runif(0,1);

      if(log(rand_unif_var) < acceptance_prob)
      {
        // Accept new state and update parameters
        a(i, iter) = new_a;
      }
    }else
    {
      a_lpdf = lpdf_a2(alpha_2l, beta_2l, a(i, iter), delta);
      new_a = r_truncnorm(a(i, iter), var_epsilon1 / beta_2l, 0,
                          std::numeric_limits<double>::infinity());

      a_new_lpdf = lpdf_a2(alpha_2l, beta_2l, new_a, delta);

      acceptance_prob = (a_new_lpdf + d_truncnorm(a(i, iter), new_a,
                                                  var_epsilon1 / beta_2l, 0,
                                                  std::numeric_limits<double>::infinity(),
                                                  1)) - a_lpdf -
                                                    d_truncnorm(new_a, a(i, iter),
                                                                var_epsilon1 / beta_2l, 0,
                                                                std::numeric_limits<double>::infinity(),
                                                                1);
      rand_unif_var = R::runif(0,1);

      if(log(rand_unif_var) < acceptance_prob)
      {
        // Accept new state and update parameters
        a(i, iter) = new_a;
      }
    }

  }
}
