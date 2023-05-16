#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include <BayesFMMM.h>

//' Find initial starting position for mean and allocation structure for functional data
//'
//' Function for finding a good initial starting point for nu parameters, Z, and
//' eta (if covariate adjusted)
//' parameters for functional data, with option for tempered transitions. This
//' function tries running multiple different MCMC chains to find the optimal
//' starting position. This function will return the chain that has the highest
//' log-likelihood average in the last 100 MCMC iterations. To specify a covariate
//' adjusted model, please provide the design matrix X.
//'
//' @name BFMMM_Nu_Z_multiple_try
//' @param tot_mcmc_iters Int containing the number of MCMC iterations per try
//' @param n_try Int containing how many different chains are tried
//' @param K Int containing the number of clusters
//' @param Y List of vectors containing the observed values
//' @param time List of vectors containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param basis_degree Int containing the degree of B-splines used
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param c Vector containing hyperparmeters for sampling from pi (If left NULL, the one vector will be used)
//' @param b Double containing hyperparameter for sampling from alpha_3
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparamete for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @returns a List containing:
//' \describe{
//'   \item{\code{B}}{The basis functions evaluated at the observed time points}
//'   \item{\code{nu}}{Nu samples from the chain with the highest average log-likelihood}
//'   \item{\code{eta}}{Eta samples from the chain with the highest average log-likelihood (if covariate adjusted)}
//'   \item{\code{pi}}{Pi samples from the chain with the highest average log-likelihood}
//'   \item{\code{alpha_3}}{Alpha_3 samples from the chain with the highest average log-likelihood}
//'   \item{\code{A}}{A samples from the chain with the highest average log-likelihood}
//'   \item{\code{delta}}{Delta samples from the chain with the highest average log-likelihood}
//'   \item{\code{sigma_sq}}{sigma_sq samples from the chain with the highest average log-likelihood}
//'   \item{\code{tau}}{Tau samples from the chain with the highest average log-likelihood}
//'   \item{\code{tau_eta}}{Tau_Eta samples from the chain with the highest average log-likelihood (if covariate adjusted)}
//'   \item{\code{Z}}{Z samples from the chain with the highest average log-likelihood}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//'
//' @section Warning:
//' The following must be true:
//' //' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{n_try}}{must be an integer larger than or equal to 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_funct}}{must be an integer larger than 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{X}}{must have n_funct number of rows (if specified)}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha_nu}}{must be positive}
//'   \item{\code{beta_nu}}{must be positive}
//'   \item{\code{alpha_eta}}{must be positive}
//'   \item{\code{beta_eta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 40
//' basis_degree <- 3
//' n_eigen <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' ## Run function
//' x <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                              basis_degree, n_eigen, boundary_knots,
//'                              internal_knots)
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 40
//' basis_degree <- 3
//' n_eigen <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                 basis_degree, n_eigen, boundary_knots,
//'                                 internal_knots, X = X)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List BFMMM_Nu_Z_multiple_try(const int tot_mcmc_iters,
                                   const int n_try,
                                   const int K,
                                   const arma::field<arma::vec> Y,
                                   const arma::field<arma::vec> time,
                                   const int n_funct,
                                   const int basis_degree,
                                   const int n_eigen,
                                   const arma::vec boundary_knots,
                                   const arma::vec internal_knots,
                                   Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                                   Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                                   const double b = 10,
                                   const double nu_1 = 3,
                                   const double alpha1l = 1,
                                   const double alpha2l= 2,
                                   const double beta1l = 1,
                                   const double beta2l = 1,
                                   const double a_Z_PM = 10000,
                                   const double a_pi_PM = 1000,
                                   const double var_alpha3 = 0.05,
                                   const double var_epsilon1 = 1,
                                   const double var_epsilon2 = 1,
                                   const double alpha_nu = 10,
                                   const double beta_nu = 1,
                                   const double alpha_eta = 10,
                                   const double beta_eta = 1,
                                   const double alpha_0 = 1,
                                   const double beta_0 = 1){
  Rcpp::List BestChain;

  if(X.isNull()){
    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }

    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    //Create B-splines
    splines2::BSpline bspline;
    // Make B_obs
    arma::field<arma::mat> B_obs(n_funct,1);
    for(int i = 0; i < n_funct; i++){
      // Create Bspline object
      bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix (100 x 8)
      arma::mat bspline_mat{bspline.basis(true)};
      B_obs(i,0) = bspline_mat;
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_Nu_Z(Y, time, n_funct, K, basis_degree,
                                            n_eigen, boundary_knots, internal_knots,
                                            tot_mcmc_iters, c1, b, alpha1l,
                                            alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                            var_alpha3, var_epsilon1, var_epsilon2,
                                            alpha_nu, beta_nu, alpha_0, beta_0);
    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BFMMM_Nu_Z(Y, time, n_funct, K, basis_degree,
                                              n_eigen, boundary_knots, internal_knots,
                                              tot_mcmc_iters, c1, b, alpha1l,
                                              alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                              var_alpha3, var_epsilon1, var_epsilon2,
                                              alpha_nu, beta_nu, alpha_0, beta_0);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("B", B_obs),
                                    Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("pi", mod1["pi"]),
                                    Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(X1.n_rows != n_funct){
      Rcpp::stop("'X' must be have 'n_funct' number of rows");
    }
    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }

    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha_eta' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta_eta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    //Create B-splines
    splines2::BSpline bspline;
    // Make B_obs
    arma::field<arma::mat> B_obs(n_funct,1);
    for(int i = 0; i < n_funct; i++){
      // Create Bspline object
      bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix (100 x 8)
      arma::mat bspline_mat{bspline.basis(true)};
      B_obs(i,0) = bspline_mat;
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_Nu_Z_Cov_Adj(Y, time, X1, n_funct, K, basis_degree,
                                                    n_eigen, boundary_knots, internal_knots,
                                                    tot_mcmc_iters, c1, b, alpha1l,
                                                    alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                    var_alpha3, var_epsilon1, var_epsilon2,
                                                    alpha_nu, beta_nu, alpha_eta, beta_eta,
                                                    alpha_0, beta_0);
    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi =BayesFMMM::BFMMM_Nu_Z_Cov_Adj(Y, time, X1, n_funct, K, basis_degree,
                                                     n_eigen, boundary_knots, internal_knots,
                                                     tot_mcmc_iters, c1, b, alpha1l,
                                                     alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                     var_alpha3, var_epsilon1, var_epsilon2,
                                                     alpha_nu, beta_nu, alpha_eta, beta_eta,
                                                     alpha_0, beta_0);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("B", B_obs),
                                    Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("eta", mod1["eta"]),
                                    Rcpp::Named("pi", mod1["pi"]),
                                    Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                    Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }


  return BestChain;
}

//' Find initial starting points for covariance parameters given mean and allocation structure for functional data
//'
//' This function is meant to be used after using \code{BFMMM_NU_Z_multiple_try}.
//' This function samples from the rest of the model parameters given a fixed value of
//' nu and Z. The fixed value of nu, Z, and eta (if covariate adjusted) are found by using the best markov chain
//' found in \code{BFMMM_NU_Z_multiple_try}. Once this function is ran, the results
//' can be used in \code{BFMMM_warm_start}. To specify a covariate
//' adjusted model, please provide the design matrix X.
//'
//' @name BFMMM_Theta_est
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param n_try Int containing how many different chains are tried
//' @param K Int containing the number of clusters
//' @param Y List of vectors containing the observed values
//' @param time List of vectors containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param basis_degree Int containing the degree of B-splines used
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param multiple_try List containing results from \code{BFMMM_Nu_Z_multiple_try}
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param c Vector containing hyperparmeter for sampling from pi (If left NULL, the one vector will be used)
//' @param b double containing hyperparamete for sampling from alpha_3
//' @param nu_1 double containing hyperparameter for sampling from gamma
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @param covariance_adj Boolean containing whether or not covariates should affect the covariance
//' @returns a List containing:
//' \describe{
//'   \item{\code{B}}{The basis functions evaluated at the observed time points}
//'   \item{\code{Z}}{estimates of Z}
//'   \item{\code{nu}}{estimates of nu}
//'   \item{\code{eta}}{estimated value from \code{BFMMM_Nu_Z_multiple_try} (if covariate adjusted)}
//'   \item{\code{chi}}{chi samples from MCMC chain}
//'   \item{\code{A}}{A samples from MCMC chain}
//'   \item{\code{A_xi}}{A_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{delta}}{delta samples from MCMC chain}
//'   \item{\code{delta_xi}}{delta_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{gamma_xi}}{gamma_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{sigma_sq}}{sigma_sq samples from MCMC chain}
//'   \item{\code{tau}}{tau samples from MCMC chain}
//'   \item{\code{tau_eta}}{tau_eta samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{gamma}}{gamma samples from the MCMC chain}
//'   \item{\code{Phi}}{Phi samples from MCMC chain}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{n_try}}{must be an integer larger than or equal to 1}
//'   \item{\code{burnin_prop}}{must be between 0 and 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_funct}}{must be an integer larger than 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{X}}{must have n_funct number of rows (if specified)}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{nu_1}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha_nu}}{must be positive}
//'   \item{\code{beta_nu}}{must be positive}
//'   \item{\code{alpha_eta}}{must be positive}
//'   \item{\code{beta_eta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 40
//' basis_degree <- 3
//' n_eigen <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                 basis_degree, n_eigen, boundary_knots,
//'                                 internal_knots)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                         basis_degree, n_eigen, boundary_knots,
//'                         internal_knots, est1)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 40
//' basis_degree <- 3
//' n_eigen <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                 basis_degree, n_eigen, boundary_knots,
//'                                 internal_knots, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                         basis_degree, n_eigen, boundary_knots,
//'                         internal_knots, est1, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 40
//' basis_degree <- 3
//' n_eigen <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                 basis_degree, n_eigen, boundary_knots,
//'                                 internal_knots, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                         basis_degree, n_eigen, boundary_knots,
//'                         internal_knots, est1, X = X, covariance_adj = T)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List BFMMM_Theta_est(const int tot_mcmc_iters,
                           const int n_try,
                           const int K,
                           const arma::field<arma::vec> Y,
                           const arma::field<arma::vec> time,
                           const int n_funct,
                           const int basis_degree,
                           const int n_eigen,
                           const arma::vec boundary_knots,
                           const arma::vec internal_knots,
                           const Rcpp::List multiple_try,
                           Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                           const double burnin_prop = 0.8,
                           Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                           const double b = 10,
                           const double nu_1 = 3,
                           const double alpha1l = 2,
                           const double alpha2l = 3,
                           const double beta1l = 2,
                           const double beta2l = 2,
                           const double a_Z_PM = 10000,
                           const double a_pi_PM = 1000,
                           const double var_alpha3 = 0.05,
                           const double var_epsilon1 = 1,
                           const double var_epsilon2 = 1,
                           const double alpha_nu = 10,
                           const double beta_nu = 1,
                           const double alpha_eta = 10,
                           const double beta_eta = 1,
                           const double alpha_0 = 1,
                           const double beta_0 = 1,
                           const bool covariance_adj = false){
  Rcpp::List BestChain;

  if(X.isNull()){
    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    splines2::BSpline bspline;
    // Make B_obs
    arma::field<arma::mat> B_obs(n_funct,1);
    for(int i = 0; i < n_funct; i++)
    {
      // Create Bspline object
      bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix (100 x 8)
      arma::mat bspline_mat{bspline.basis(true)};
      B_obs(i,0) = bspline_mat;
    }

    int n_nu = nu_samp.n_slices;
    arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      for(int j = 0; j < Z_est.n_cols; j++){
        Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_Theta(Y, time, n_funct, K, basis_degree, n_eigen,
                                             boundary_knots, internal_knots, tot_mcmc_iters,
                                             c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                             beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                             var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                             alpha_0, beta_0, Z_est, nu_est);

    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BFMMM_Theta(Y, time, n_funct, K, basis_degree, n_eigen,
                                               boundary_knots, internal_knots, tot_mcmc_iters,
                                               c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                               beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                               var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                               alpha_0, beta_0, Z_est, nu_est);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("B", B_obs),
                                    Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("chi", mod1["chi"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("gamma", mod1["gamma"]),
                                    Rcpp::Named("Phi", mod1["Phi"]),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::field<arma::cube> eta_samp = multiple_try["eta"];

    if(eta_samp(0,0).n_rows != nu_samp.n_cols){
      Rcpp::stop("The specified 'Eta' is of the wrong dimension");
    }
    if(X1.n_rows != n_funct){
      Rcpp::stop("'X' must be have 'n_funct' number of rows");
    }
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha_eta' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta_eta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    splines2::BSpline bspline;
    // Make B_obs
    arma::field<arma::mat> B_obs(n_funct,1);
    for(int i = 0; i < n_funct; i++)
    {
      // Create Bspline object
      bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix (100 x 8)
      arma::mat bspline_mat{bspline.basis(true)};
      B_obs(i,0) = bspline_mat;
    }

    int n_nu = nu_samp.n_slices;
    arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::cube eta_est = arma::zeros(eta_samp(0,0).n_rows, eta_samp(0,0).n_cols, eta_samp(0,0).n_slices);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
        for(int d = 0; d < X1.n_cols; d++){
          for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
            ph_eta(l - std::round(n_nu * burnin_prop)) = eta_samp(l,0)(j,d,i);
          }
          eta_est(j,d,i) = arma::median(ph_eta);
        }
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      for(int j = 0; j < Z_est.n_cols; j++){
        Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_Theta_Cov_Adj(Y, time, X1, n_funct, K, basis_degree, n_eigen,
                                                     boundary_knots, internal_knots, tot_mcmc_iters,
                                                     c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                                     beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                                     var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                                     alpha_eta, beta_eta, alpha_0, beta_0, Z_est, nu_est,
                                                     eta_est, covariance_adj);

    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BFMMM_Theta_Cov_Adj(Y, time, X1, n_funct, K, basis_degree, n_eigen,
                                                       boundary_knots, internal_knots, tot_mcmc_iters,
                                                       c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                                       beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                                       var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                                       alpha_eta, beta_eta, alpha_0, beta_0, Z_est, nu_est,
                                                       eta_est, covariance_adj);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain = Rcpp::List::create(Rcpp::Named("B", B_obs),
                                   Rcpp::Named("Z", mod1["Z"]),
                                   Rcpp::Named("nu", mod1["nu"]),
                                   Rcpp::Named("eta", mod1["eta"]),
                                   Rcpp::Named("chi", mod1["chi"]),
                                   Rcpp::Named("A", mod1["A"]),
                                   Rcpp::Named("A_xi", mod1["A_xi"]),
                                   Rcpp::Named("delta", mod1["delta"]),
                                   Rcpp::Named("delta_xi", mod1["delta_xi"]),
                                   Rcpp::Named("gamma_xi", mod1["gamma_xi"]),
                                   Rcpp::Named("sigma_sq", mod1["sigma"]),
                                   Rcpp::Named("tau", mod1["tau"]),
                                   Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                   Rcpp::Named("gamma", mod1["gamma"]),
                                   Rcpp::Named("Phi", mod1["Phi"]),
                                   Rcpp::Named("xi", mod1["xi"]),
                                   Rcpp::Named("loglik", mod1["loglik"]));

  }

  return BestChain;
}

//' Performs MCMC for functional models given an informed set of starting points
//'
//' This function is meant to be used after using \code{BFMMM_Nu_Z_multiple_try}
//' and \code{BFMMM_Theta_est}. This function will use the outputs of these two
//' functions to start the MCMC chain in a good location. Since the posterior distribution
//' can often be multimodal, it is important to have a good starting position.
//' To help move across modes, this function allows users to use tempered transitions
//' every \code{n_temp_trans} iterations. By using a mixture of tempered transitions
//' and un-tempered transitions, we can allow the chain to explore multiple modes without
//' while keeping sampling relatively computationally efficient. To save on RAM usage, we
//' allow users to specify how many samples are kept in memory using \code{r_stored_iters}.
//' If \code{r_stored_iters} is less than \code{tot_mcmc_iters}, then a thinned version
//' of the chain is stored in the user specified directory (\code{dir}). The samples from each
//' parameter can be viewed using the following functions: \code{ReadFieldCube},
//' \code{ReadFieldMat}, \code{ReadFieldVec}, \code{ReadCube}, \code{ReadMat},
//' \code{ReadVec}. To specify a covariate
//' adjusted model, please provide the design matrix X.
//'
//' @name BFMMM_warm_start
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param K Int containing the number of clusters
//' @param Y List of vectors containing the observed values
//' @param time List of vectors containing the observed time points
//' @param n_funct Int containing the number of functions
//' @param basis_degree Int containing the degree of B-splines used
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param multiple_try List containing results from \code{BFMMM_Nu_Z_multiple_try}
//' @param theta_est List containing results from \code{BFMMM_Theta_est}
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param dir String containing directory where the MCMC files should be saved (if NULL, then no files will be saved)
//' @param thinning_num Int containing how often we should save MCMC iterations
//' @param beta_N_t Double containing the maximum weight for tempered transitions
//' @param N_t Int containing total number of tempered transitions
//' @param n_temp_trans Int containing how often tempered transitions are performed (if 0, then no tempered transitions are performed)
//' @param r_stored_iters Int containing how many MCMC iterations are stored in RAM (if 0, then all MCMC iterations are stored in RAM)
//' @param c Vector containing hyperparmeter for sampling from pi (If left NULL, the one vector will be used)
//' @param b double containing hyperparamete for sampling from alpha_3
//' @param nu_1 double containing hyperparameter for sampling from gamma
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @param covariance_adj Boolean containing whether or not the covariance structure should depend on the covariates
//'
//' @returns a List containing:
//' \describe{
//'   \item{\code{B}}{The basis functions evaluated at the observed time points}
//'   \item{\code{nu}}{Nu samples from the MCMC chain}
//'   \item{\code{chi}}{chi samples from the MCMC chain}
//'   \item{\code{pi}}{pi samples from the MCMC chain}
//'   \item{\code{alpha_3}}{alpha_3 samples from the MCMC chain}
//'   \item{\code{A}}{A samples from MCMC chain}
//'   \item{\code{delta}}{delta samples from the MCMC chain}
//'   \item{\code{sigma_sq}}{sigma_sq samples from the MCMC chain}
//'   \item{\code{tau}}{tau samples from the MCMC chain}
//'   \item{\code{tau_eta}}{tau_eta samples from the MCMC chain (if covariate adjusted)}
//'   \item{\code{eta}}{eta samples from the MCMC chain (if covariate adjusted)}
//'   \item{\code{xi}}{xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{delta_xi}}{delta_xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{gamma_xi}}{gamma_xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{A_xi}}{A_xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{gamma}}{gamma samples from the MCMC chain}
//'   \item{\code{Phi}}{Phi samples from the MCMC chain}
//'   \item{\code{Z}}{Z samples from the MCMC chain}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{X}}{ must have n_funct number of rows}
//'   \item{\code{burnin_prop}}{must be between 0 and 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_funct}}{must be an integer larger than 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{X}}{must have n_funct number of rows (if specified)}
//'   \item{\code{dir}}{must be specified if \code{r_stored_iters} <= \code{tot_mcmc_iters} (other than if \code{r_stored_iters} = 0)}
//'   \item{\code{n_thinning}}{must be a positive integer}
//'   \item{\code{beta_N_t}}{must be between 1 and 0}
//'   \item{\code{N_t}}{must be a positive integer}
//'   \item{\code{n_temp_trans}}{must be a non-negative integer}
//'   \item{\code{r_stored_iters}}{must be a non-negative integer}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{nu_1}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha_nu}}{must be positive}
//'   \item{\code{beta_nu}}{must be positive}
//'   \item{\code{alpha_eta}}{must be positive}
//'   \item{\code{beta_eta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//'@examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 40
//' basis_degree <- 3
//' n_eigen <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                 basis_degree, n_eigen, boundary_knots,
//'                                 internal_knots)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                         basis_degree, n_eigen, boundary_knots,
//'                         internal_knots, est1)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BFMMM_warm_start(tot_mcmc_iters, K, Y, time, n_funct,
//'                                basis_degree, n_eigen, boundary_knots,
//'                                internal_knots, est1, est2)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 40
//' basis_degree <- 3
//' n_eigen <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                 basis_degree, n_eigen, boundary_knots,
//'                                 internal_knots, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                         basis_degree, n_eigen, boundary_knots,
//'                         internal_knots, est1, X = X)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BFMMM_warm_start(tot_mcmc_iters, K, Y, time, n_funct,
//'                                basis_degree, n_eigen, boundary_knots,
//'                                internal_knots, est1, est2, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 40
//' basis_degree <- 3
//' n_eigen <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                 basis_degree, n_eigen, boundary_knots,
//'                                 internal_knots, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                         basis_degree, n_eigen, boundary_knots,
//'                         internal_knots, est1, X = X, covariance_adj = T)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BFMMM_warm_start(tot_mcmc_iters, K, Y, time, n_funct,
//'                                basis_degree, n_eigen, boundary_knots,
//'                                internal_knots, est1, est2, X = X, covariance_adj = T)
//' @export
// [[Rcpp::export]]
Rcpp::List BFMMM_warm_start(const int tot_mcmc_iters,
                            const int K,
                            const arma::field<arma::vec> Y,
                            const arma::field<arma::vec> time,
                            const int n_funct,
                            const int basis_degree,
                            const int n_eigen,
                            const arma::vec boundary_knots,
                            const arma::vec internal_knots,
                            const Rcpp::List multiple_try,
                            const Rcpp::List theta_est,
                            Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                            const double burnin_prop = 0.8,
                            Rcpp::Nullable<Rcpp::CharacterVector> dir = R_NilValue,
                            const double thinning_num = 1,
                            const double beta_N_t = 1,
                            int N_t = 1,
                            int n_temp_trans = 0,
                            int r_stored_iters = 0,
                            Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                            const double b = 10,
                            const double nu_1 = 3,
                            const double alpha1l = 2,
                            const double alpha2l = 3,
                            const double beta1l = 2,
                            const double beta2l = 2,
                            const double a_Z_PM = 10000,
                            const double a_pi_PM = 1000,
                            const double var_alpha3 = 0.05,
                            const double var_epsilon1 = 1,
                            const double var_epsilon2 = 1,
                            const double alpha_nu = 10,
                            const double beta_nu = 1,
                            const double alpha_eta = 10,
                            const double beta_eta = 1,
                            const double alpha_0 = 1,
                            const double beta_0 = 1,
                            const bool covariance_adj = false){
  Rcpp::List mod2;

  if(X.isNull()){
    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::mat pi_samp = multiple_try["pi"];
    arma::vec alpha_3_samp = multiple_try["alpha_3"];
    arma::cube delta_samp = theta_est["delta"];
    arma::field<arma::cube> gamma_samp = theta_est["gamma"];
    arma::field<arma::cube> Phi_samp = theta_est["Phi"];
    arma::cube A_samp = theta_est["A"];
    arma::mat tau_samp = multiple_try["tau"];
    arma::vec sigma_samp = theta_est["sigma_sq"];
    arma::cube chi_samp = theta_est["chi"];

    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }
    if(thinning_num <= 0){
      Rcpp::stop("'thinning_num' must be a positive integer");
    }
    if(beta_N_t <= 0){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(beta_N_t > 1){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(N_t < 1){
      Rcpp::stop("'N_t' must be a positive integer");
    }
    if(r_stored_iters < 0){
      Rcpp::stop("'r_stored_iters' must be a non-negative integer");
    }
    if(n_temp_trans < 0){
      Rcpp::stop("'n_temp_trans' must be a non-negative integer");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()){
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    // if r_stored_iters is default, do not save anything
    std::string dir1 = "";
    if(r_stored_iters == 0){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // check if directory is specified
    if(dir.isNotNull()){
      Rcpp::CharacterVector s(dir);
      dir1 = std::string(s[0]);

      // save entire chain at last iteration
      if(r_stored_iters == 0){
        r_stored_iters = tot_mcmc_iters;
      }
    }

    // Check if there is a place to store files if r_stored_iters < tot_mcmc_iters
    if(dir.isNull()){
      if(r_stored_iters <= tot_mcmc_iters){
        Rcpp::stop("'r_stored_iters' <= 'tot_mcmc_iters' with no 'dir' specified. Either specify 'dir' or increase 'r_stored_iters'");
      }
    }

    // if n_temp_trans is default set to greater than tot_mcmc_iters
    if(n_temp_trans == 0){
      n_temp_trans = tot_mcmc_iters + 1;
      N_t = 1;
    }

    // save RAM
    if(r_stored_iters > tot_mcmc_iters + 1){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // Start of Algorithm
    splines2::BSpline bspline;
    // Make B_obs
    arma::field<arma::mat> B_obs(n_funct,1);
    for(int i = 0; i < n_funct; i++)
    {
      // Create Bspline object
      bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix (100 x 8)
      arma::mat bspline_mat{bspline.basis(true)};
      B_obs(i,0) = bspline_mat;
    }

    int n_nu = alpha_3_samp.n_elem;

    double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    arma::vec pi_est = arma::zeros(pi_samp.n_rows);
    arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < Z_est.n_cols; i++){
      pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
    }

    pi_est = pi_est / arma::accu(pi_est);

    int n_Phi = sigma_samp.n_elem;

    double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
    arma::mat delta_est = arma::zeros(delta_samp.n_rows, delta_samp.n_cols);
    arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < delta_samp.n_cols; j++){
      for(int i = 0; i < delta_samp.n_rows; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,j,l);
        }
        delta_est(i, j) = arma::median(ph_delta);
      }
    }

    arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
    arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
    arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < Phi_est.n_rows; i++){
      for(int j = 0; j < Phi_est.n_cols; j++){
        for(int m = 0; m < Phi_est.n_slices; m++){
          for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
            ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);

            ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
          }
          Phi_est(i,j,m) = arma::median(ph_phi);
          gamma_est(i,j,m) = arma::median(ph_gamma);
        }
      }
    }

    arma::mat A_est = arma::zeros(A_samp.n_rows, A_samp.n_cols);
    arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < A_samp.n_rows; j++){
      for(int i = 0; i < A_samp.n_cols; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(j, i, l);
        }
        A_est(j, i) = arma::median(ph_A);
      }
    }

    arma::vec tau_est = arma::zeros(tau_samp.n_cols);
    arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < tau_est.n_elem; i++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
      }
      tau_est(i) = arma::median(ph_tau);
    }
    arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
    arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < chi_est.n_rows; i++){
      for(int j = 0; j < chi_est.n_cols; j++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
        }
        chi_est(i,j) = arma::median(ph_chi);
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_MTT_warm_start(Y, time, n_funct, thinning_num, K,
                                                      basis_degree, n_eigen, boundary_knots,
                                                      internal_knots, tot_mcmc_iters,
                                                      r_stored_iters, n_temp_trans,
                                                      c1, b, nu_1, alpha1l, alpha2l,
                                                      beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                      var_alpha3, var_epsilon1,
                                                      var_epsilon2, alpha_nu, beta_nu, alpha_0,
                                                      beta_0, dir1, beta_N_t, N_t,
                                                      Z_est, pi_est, alpha_3_est,
                                                      delta_est, gamma_est, Phi_est, A_est,
                                                      nu_est, tau_est, sigma_est, chi_est);

    mod2 =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                               Rcpp::Named("Z", mod1["Z"]),
                               Rcpp::Named("nu", mod1["nu"]),
                               Rcpp::Named("chi", mod1["chi"]),
                               Rcpp::Named("pi", mod1["pi"]),
                               Rcpp::Named("alpha_3", mod1["alpha_3"]),
                               Rcpp::Named("A", mod1["A"]),
                               Rcpp::Named("delta", mod1["delta"]),
                               Rcpp::Named("sigma_sq", mod1["sigma"]),
                               Rcpp::Named("tau", mod1["tau"]),
                               Rcpp::Named("gamma", mod1["gamma"]),
                               Rcpp::Named("Phi", mod1["Phi"]),
                               Rcpp::Named("Z", mod1["Z"]),
                               Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::mat pi_samp = multiple_try["pi"];
    arma::vec alpha_3_samp = multiple_try["alpha_3"];
    arma::cube delta_samp = theta_est["delta"];
    arma::field<arma::cube> gamma_samp = theta_est["gamma"];
    arma::field<arma::cube> Phi_samp = theta_est["Phi"];
    arma::cube A_samp = theta_est["A"];
    arma::mat tau_samp = multiple_try["tau"];
    arma::vec sigma_samp = theta_est["sigma_sq"];
    arma::cube chi_samp = theta_est["chi"];
    arma::field<arma::cube> eta_samp = multiple_try["eta"];
    arma::cube tau_eta_samp = multiple_try["tau_eta"];

    // generate warnings
    if(X1.n_rows != n_funct){
      Rcpp::stop("'X' must be have 'n_funct' number of rows");
    }
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }
    if(thinning_num <= 0){
      Rcpp::stop("'thinning_num' must be a positive integer");
    }
    if(beta_N_t <= 0){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(beta_N_t > 1){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(N_t < 1){
      Rcpp::stop("'N_t' must be a positive integer");
    }
    if(r_stored_iters < 0){
      Rcpp::stop("'r_stored_iters' must be a non-negative integer");
    }
    if(n_temp_trans < 0){
      Rcpp::stop("'n_temp_trans' must be a non-negative integer");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()){
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    // if r_stored_iters is default, do not save anything
    std::string dir1 = "";
    if(r_stored_iters == 0){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // check if directory is specified
    if(dir.isNotNull()){
      Rcpp::CharacterVector s(dir);
      dir1 = std::string(s[0]);

      // save entire chain at last iteration
      if(r_stored_iters == 0){
        r_stored_iters = tot_mcmc_iters;
      }
    }

    // Check if there is a place to store files if r_stored_iters < tot_mcmc_iters
    if(dir.isNull()){
      if(r_stored_iters <= tot_mcmc_iters){
        Rcpp::stop("'r_stored_iters' <= 'tot_mcmc_iters' with no 'dir' specified. Either specify 'dir' or increase 'r_stored_iters'");
      }
    }

    // if n_temp_trans is default set to greater than tot_mcmc_iters
    if(n_temp_trans == 0){
      n_temp_trans = tot_mcmc_iters + 1;
      N_t = 1;
    }

    // save RAM
    if(r_stored_iters > tot_mcmc_iters + 1){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // Start of Algorithm
    splines2::BSpline bspline;
    // Make B_obs
    arma::field<arma::mat> B_obs(n_funct,1);
    for(int i = 0; i < n_funct; i++)
    {
      // Create Bspline object
      bspline = splines2::BSpline(time(i,0), internal_knots, basis_degree,
                                  boundary_knots);
      // Get Basis matrix (100 x 8)
      arma::mat bspline_mat{bspline.basis(true)};
      B_obs(i,0) = bspline_mat;
    }

    int n_nu = alpha_3_samp.n_elem;

    double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    arma::vec pi_est = arma::zeros(pi_samp.n_rows);
    arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::cube eta_est = arma::zeros(eta_samp(0,0).n_rows, eta_samp(0,0).n_cols, eta_samp(0,0).n_slices);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

    for(int i = 0; i < Z_est.n_cols; i++){
      pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    }

    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
        for(int d = 0; d < X1.n_cols; d++){
          for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
            ph_eta(l - std::round(n_nu * burnin_prop)) = eta_samp(l,0)(j,d,i);
          }
          eta_est(j,d,i) = arma::median(ph_eta);
        }
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
    }

    pi_est = pi_est / arma::accu(pi_est);

    int n_Phi = sigma_samp.n_elem;

    double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
    arma::mat delta_est = arma::zeros(delta_samp.n_rows, delta_samp.n_cols);
    arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < delta_samp.n_cols; j++){
      for(int i = 0; i < delta_samp.n_rows; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,j,l);
        }
        delta_est(i, j) = arma::median(ph_delta);
      }
    }

    arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
    arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
    arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < Phi_est.n_rows; i++){
      for(int j = 0; j < Phi_est.n_cols; j++){
        for(int m = 0; m < Phi_est.n_slices; m++){
          for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
            ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);

            ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
          }
          Phi_est(i,j,m) = arma::median(ph_phi);
          gamma_est(i,j,m) = arma::median(ph_gamma);
        }
      }
    }

    arma::mat A_est = arma::zeros(A_samp.n_rows, A_samp.n_cols);
    arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < A_samp.n_rows; j++){
      for(int i = 0; i < A_samp.n_cols; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(j, i, l);
        }
        A_est(j, i) = arma::median(ph_A);
      }
    }

    arma::vec tau_est = arma::zeros(tau_samp.n_cols);
    arma::mat tau_eta_est = arma::zeros(tau_eta_samp.n_rows, tau_eta_samp.n_cols);
    arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_tau_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < tau_est.n_elem; i++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
      }
      tau_est(i) = arma::median(ph_tau);
      for(int j = 0; j < tau_eta_est.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_tau_eta(l - std::round(n_nu * burnin_prop)) = tau_eta_samp(i,j,l);
        }
        tau_eta_est(i,j) = arma::median(ph_tau_eta);
      }
    }
    arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
    arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < chi_est.n_rows; i++){
      for(int j = 0; j < chi_est.n_cols; j++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
        }
        chi_est(i,j) = arma::median(ph_chi);
      }
    }

    if(covariance_adj == false){
      // start MCMC sampling
      Rcpp::List mod1 = BayesFMMM::BFMMM_MTT_warm_start_MeanAdj(Y, time, X1, n_funct, thinning_num, K,
                                                                basis_degree, n_eigen, boundary_knots,
                                                                internal_knots, tot_mcmc_iters,
                                                                r_stored_iters, n_temp_trans,
                                                                c1, b, nu_1, alpha1l, alpha2l,
                                                                beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                                var_alpha3, var_epsilon1,
                                                                var_epsilon2, alpha_nu, beta_nu,
                                                                alpha_eta, beta_eta, alpha_0,
                                                                beta_0, dir1, beta_N_t, N_t,
                                                                Z_est, pi_est, alpha_3_est,
                                                                delta_est, gamma_est, Phi_est, A_est,
                                                                nu_est, eta_est, tau_est, tau_eta_est,
                                                                sigma_est, chi_est);
      mod2 =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                 Rcpp::Named("nu", mod1["nu"]),
                                 Rcpp::Named("chi", mod1["chi"]),
                                 Rcpp::Named("pi", mod1["pi"]),
                                 Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                 Rcpp::Named("A", mod1["A"]),
                                 Rcpp::Named("delta", mod1["delta"]),
                                 Rcpp::Named("sigma_sq", mod1["sigma"]),
                                 Rcpp::Named("tau", mod1["tau"]),
                                 Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                 Rcpp::Named("eta", mod1["eta"]),
                                 Rcpp::Named("gamma", mod1["gamma"]),
                                 Rcpp::Named("Phi", mod1["Phi"]),
                                 Rcpp::Named("Z", mod1["Z"]),
                                 Rcpp::Named("loglik", mod1["loglik"]));
    }else{
      arma::field<arma::cube> delta_xi_samp = theta_est["delta_xi"];
      arma::field<arma::cube> A_xi_samp = theta_est["A_xi"];
      arma::field<arma::cube> xi_samp1 = theta_est["xi"];
      arma::field<arma::cube> xi_samp(n_nu, K);
      for(int k = 0; k < K; k++){
        for(int n = 0; n < n_nu; n++){
          xi_samp(n,k) = xi_samp1(((k * n_nu) + n), 0);
        }
      }
      arma::field<arma::cube> gamma_xi_samp1 = theta_est["gamma_xi"];
      arma::field<arma::cube> gamma_xi_samp(n_nu, K);
      for(int k = 0; k < K; k++){
        for(int n = 0; n < n_nu; n++){
          gamma_xi_samp(n,k) = gamma_xi_samp1(((k * n_nu) + n), 0);
        }
      }

      arma::cube delta_xi_est = arma::zeros(delta_xi_samp(0,0).n_rows,
                                            delta_xi_samp(0,0).n_cols,
                                            delta_xi_samp(0,0).n_slices);
      arma::cube A_xi_est = arma::zeros(A_xi_samp(0,0).n_rows,
                                        A_xi_samp(0,0).n_cols,
                                        A_xi_samp(0,0).n_slices);

      arma::field<arma::cube> xi_est(1,K);
      arma::field<arma::cube> gamma_xi_est(1,K);

      for(int j = 0; j < K; j++){
        Rcpp::Rcout << "j =" << K;
        xi_est(0,j) = arma::zeros(xi_samp(0,j).n_rows, xi_samp(0,j).n_cols,
               xi_samp(0,j).n_slices);
        gamma_xi_est(0,j) = arma::zeros(gamma_xi_samp(0,j).n_rows,
                     gamma_xi_samp(0,j).n_cols, gamma_xi_samp(0,j).n_slices);
      }

      arma::vec delta_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec A_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec gamma_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

      for(int j = 0; j < K; j++){
        for(int d = 0; d < X1.n_cols; d++){
          for(int i = 0; i < 2; i++){
            for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
              A_xi_ph(l - std::round(n_nu * burnin_prop)) = A_xi_samp(l,0)(j, i, d);
            }
            A_xi_est(j, i, d) = arma::median(A_xi_ph);
          }
          for(int m = 0; m < n_eigen; m++){
            for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
              delta_xi_ph(l - std::round(n_nu * burnin_prop)) = delta_xi_samp(l,0)(j, m, d);
            }
            delta_xi_est(j, m, d) = arma::median(delta_xi_ph);
            for(int p = 0; p < xi_samp(0,j).n_rows; p++){
              for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
                xi_ph(l - std::round(n_nu * burnin_prop)) = xi_samp(l,j)(p, d, m);
                gamma_xi_ph(l - std::round(n_nu * burnin_prop)) = gamma_xi_samp(l,j)(p, d, m);
              }

              xi_est(0,j)(p, d, m) = arma::median(xi_ph);
              gamma_xi_est(0,j)(p, d, m) = arma::median(gamma_xi_ph);
            }
          }
        }
      }

      Rcpp::List mod1 = BayesFMMM::BFMMM_MTT_warm_start_Mean_CovAdj(Y, time, X1, n_funct, thinning_num, K,
                                                                    basis_degree, n_eigen, boundary_knots,
                                                                    internal_knots, tot_mcmc_iters,
                                                                    r_stored_iters, n_temp_trans,
                                                                    c1, b, nu_1, alpha1l, alpha2l,
                                                                    beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                                    var_alpha3, var_epsilon1,
                                                                    var_epsilon2, alpha_nu, beta_nu,
                                                                    alpha_eta, beta_eta, alpha_0,
                                                                    beta_0, dir1, beta_N_t, N_t,
                                                                    Z_est, pi_est, alpha_3_est,
                                                                    delta_est, delta_xi_est, gamma_est,
                                                                    gamma_xi_est, Phi_est, xi_est, A_est,
                                                                    A_xi_est, nu_est, eta_est, tau_est,
                                                                    tau_eta_est, sigma_est, chi_est);

      mod2 =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                 Rcpp::Named("nu", mod1["nu"]),
                                 Rcpp::Named("chi", mod1["chi"]),
                                 Rcpp::Named("pi", mod1["pi"]),
                                 Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                 Rcpp::Named("A", mod1["A"]),
                                 Rcpp::Named("delta", mod1["delta"]),
                                 Rcpp::Named("sigma_sq", mod1["sigma"]),
                                 Rcpp::Named("tau", mod1["tau"]),
                                 Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                 Rcpp::Named("xi", mod1["xi"]),
                                 Rcpp::Named("delta_xi", mod1["delta_xi"]),
                                 Rcpp::Named("gamma_xi", mod1["gamma_xi"]),
                                 Rcpp::Named("A_xi", mod1["A_xi"]),
                                 Rcpp::Named("eta", mod1["eta"]),
                                 Rcpp::Named("gamma", mod1["gamma"]),
                                 Rcpp::Named("Phi", mod1["Phi"]),
                                 Rcpp::Named("Z", mod1["Z"]),
                                 Rcpp::Named("loglik", mod1["loglik"]));
    }
  }

  return mod2;
}


//' Reads saved parameter data (sigma_sq, alpha_3)
//'
//' Reads armadillo vector type data and returns it as a vector in R. The following
//' parameters can be read in using this function: sigma_sq and alpha_3.
//'
//' @name ReadVec
//' @param file String containing location where armadillo vector is stored
//' @returns Vec Vector containing the saved data
//'
//' @examples
//' ## set file path
//' file <- system.file("test-data", "sigma.txt", package = "BayesFMMM")
//'
//' ## Read in file
//' sigma_sq <- ReadVec(file)
//'
//' #############################################################
//' ## For reading in a group of files you can use the following:
//' #
//' ## Set number of files you want to read in
//' # n_files <- 50
//' #
//' ## Set number of samples per file
//' # n_samp <- 100
//' #
//' ## Set directory
//' # dir <- "~/sigma"
//' #
//' ## initialize placeholder
//' # sigma_sq <- rep(0, n_files * n_samp)
//' # for(i in 0:(n_files - 1)){
//' #   sigma_i <- ReadVec(paste(dir, as.character(i),".txt", sep = ""))
//' #   sigma_sq[((n_samp * i) + 1):(n_samp * (i+1))] <- sigma_i
//' #}
//' #############################################################
//'
//' @export
// [[Rcpp::export]]
arma::vec ReadVec(std::string file){
  arma::vec B;
  B.load(file);
  return B;
}

//' Reads saved parameter data (pi, A, delta, tau)
//'
//' Reads armadillo matrix type data and returns it as a matirx in R. The following
//' parameters can be read in using this function: pi, A, delta, and tau.
//'
//' @name ReadMat
//' @param file String containing location where armadillo matrix is stored
//' @returns Mat Matrix containing the saved data
//'
//' @examples
//' ## set file path
//' file <- system.file("test-data", "pi.txt", package = "BayesFMMM")
//'
//' ## Read in file
//' pi <- ReadMat(file)
//'
//' #############################################################
//' ## For reading in a group of files you can use the following:
//' #
//' ## Set number of files you want to read in
//' # n_files <- 50
//' #
//' ## Set number of samples per file
//' # n_samp <- 100
//' #
//' ## Set dim of parameter
//' # dim <- 3
//' #
//' ## Set directory
//' # dir <- "~/pi"
//' #
//' ## initialize placeholder
//' # pi <- mat(0, dim, n_files * n_samp)
//' # for(i in 0:(n_files - 1)){
//' #   pi_i <- ReadMat(paste(dir, as.character(i),".txt", sep = ""))
//' #   pi[,((n_samp * i) + 1):(n_samp * (i+1))] <- pi_i
//' #}
//' #############################################################
//'
//' @export
// [[Rcpp::export]]
arma::mat ReadMat(std::string file){
  arma::mat B;
  B.load(file);
  return B;
}

//' Reads saved parameter data (nu, chi, Z)
//'
//' Reads armadillo cube type data and returns it as an array in R. The following
//' parameters can be read in using this function: nu, chi, and Z.
//'
//' @name ReadCube
//' @param file String containing location where armadillo cube is stored
//' @returns Cube Array containing the saved data
//'
//' @examples
//' ## set file path
//' file <- system.file("test-data", "nu.txt", package = "BayesFMMM")
//'
//' ## Read in file
//' nu <- ReadCube(file)
//'
//' #############################################################
//' ## For reading in a group of files you can use the following:
//' #
//' ## Set number of files you want to read in
//' # n_files <- 50
//' #
//' ## Set number of samples per file
//' # n_samp <- 100
//' #
//' ## Set dim of parameter
//' # dim1 <- 3
//' # dim2 <- 8
//' #
//' ## Set directory
//' # dir <- "~/Nu"
//' #
//' ## initialize placeholder
//' # nu <- array(0, dim = c(dim1, dim2, n_files * n_samp))
//' # for(i in 0:(n_files - 1)){
//' #   nu_i <- ReadCube(paste(dir, as.character(i),".txt", sep = ""))
//' #   nu[,,((n_samp * i) + 1):(n_samp * (i+1))] <- nu_i
//' #}
//' #############################################################
//'
//' @export
// [[Rcpp::export]]
arma::cube ReadCube(std::string file){
  arma::cube B;
  B.load(file);
  return B;
}

//' Reads saved parameter data (gamma, Phi)
//'
//' Reads armadillo field of cubes type data and returns it as a list of arrays
//' in R. The following parameters can be read in using this function: gamma and
//' Phi.
//'
//' @name ReadFieldCube
//' @param file String containing location where armadillo field of cubes is stored
//' @returns FieldCube List of arrays containing the saved data
//'
//' @examples
//' ## set file path
//' file <- system.file("test-data", "Phi.txt", package = "BayesFMMM")
//'
//' ## Read in file
//' Phi <- ReadFieldCube(file)
//'
//' #############################################################
//' ## For reading in a group of files you can use the following:
//' #
//' ## Set number of files you want to read in
//' # n_files <- 50
//' #
//' ## Set number of samples per file
//' # n_samp <- 100
//' #
//' ## Set dim of parameter
//' # dim1 <- 3
//' # dim2 <- 8
//' # dim3 <- 2
//' #
//' ## Set directory
//' # dir <- "~/Phi"
//' #
//' ## initialize placeholder
//' # Phi <- array(0, dim = c(dim1, dim2, dim3, n_files * n_samp))
//' # for(i in 0:(n_files - 1)){
//' #   Phi_i <- ReadFieldCube(paste(dir, as.character(i),".txt", sep = ""))
//' #   Phi[,,((n_samp * i) + 1):(n_samp * (i+1))] <- Phi_i
//' #}
//' #############################################################
//'
//' @export
// [[Rcpp::export]]
arma::field<arma::cube> ReadFieldCube(std::string file){
  arma::field<arma::cube> B;
  B.load(file);
  return B;
}

//' Reads saved armadillo data
//'
//' Reads armadillo field of matrices type data and returns it as a list of matrices
//' in R.
//'
//' @name ReadFieldMat
//' @param file String containing location where armadillo field of matrices is stored
//' @returns FieldMatrix List of matrices containing the saved data
//'
//' @examples
//' ## set file path
//' file <- system.file("test-data", "fieldmat.txt", package = "BayesFMMM")
//'
//' ## Read in file
//' samp_data <- ReadFieldMat(file)
//'
//' @export
// [[Rcpp::export]]
arma::field<arma::mat> ReadFieldMat(std::string file){
  arma::field<arma::mat> B;
  B.load(file);
  return B;
}

//' Reads saved armadillo data
//'
//' Reads armadillo field of vectors type data and returns it as a list of vectors
//' in R.
//'
//' @name ReadFieldVec
//' @param file String containing location where armadillo field of vectors is stored
//' @returns FieldVec List of vectors containing the saved data
//'
//' @examples
//' ## set file path
//' file <- system.file("test-data", "fieldvec.txt", package = "BayesFMMM")
//'
//' ## Read in file
//' samp_data <- ReadFieldVec(file)
//'
//' @export
// [[Rcpp::export]]
arma::field<arma::vec> ReadFieldVec(std::string file){
  arma::field<arma::vec> B;
  B.load(file);
  return B;
}

//' Find initial starting position for mean and allocation parameters for high dimensional functional data (Domain dimension > 1)
//'
//' Function for finding a good initial starting point for nu parameters, Z
//' parameters, and eta parameters (if covariates are included) for functional
//' data, with option for tempered transitions. This
//' function was constructed to handle data in which the domain has dimension
//' greater than 1 (i.e. a surface or higher dimensional function). This
//' function tries running multiple different MCMC chains to find the optimal
//' starting position. This function will return the chain that has the highest
//' log-likelihood average in the last 100 MCMC iterations. To specify a covariate
//' adjusted model, please provide the design matrix X.
//'
//' @name BHDFMMM_Nu_Z_multiple_try
//' @param tot_mcmc_iters Int containing the number of MCMC iterations per try
//' @param n_try Int containing how many different chains are tried
//' @param K Int containing the number of clusters
//' @param Y List of vectors containing the observed values (flattened)
//' @param time List of matrices that contain the observed time points (each column is a dimension)
//' @param n_funct Int containing the number of functions
//' @param basis_degree Vector containing the desired basis degree for each dimension
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Matrix containing the boundary knots for each dimension (each row is a dimension)
//' @param internal_knots List of vectors containing the internal knots for each dimension
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param c Vector containing hyperparmeters for sampling from pi (If left NULL, the one vector will be used)
//' @param b Double containing hyperparameter for sampling from alpha_3
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparamete for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @returns a List containing:
//' \describe{
//'   \item{\code{B}}{The basis functions evaluated at the observed time points}
//'   \item{\code{nu}}{Nu samples from the chain with the highest average log-likelihood}
//'   \item{\code{eta}}{Eta samples from the chain with the highest average log-likelihood (if covariate adjusted)}
//'   \item{\code{pi}}{Pi samples from the chain with the highest average log-likelihood}
//'   \item{\code{alpha_3}}{Alpha_3 samples from the chain with the highest average log-likelihood}
//'   \item{\code{A}}{A samples from the chain with the highest average log-likelihood}
//'   \item{\code{delta}}{Delta samples from the chain with the highest average log-likelihood}
//'   \item{\code{sigma_sq}}{sigma_sq samples from the chain with the highest average log-likelihood}
//'   \item{\code{tau}}{Tau samples from the chain with the highest average log-likelihood}
//'   \item{\code{tau_eta}}{Tau_Eta samples from the chain with the highest average log-likelihood (if covariate adjusted)}
//'   \item{\code{Z}}{Z samples from the chain with the highest average log-likelihood}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{n_try}}{must be an integer larger than or equal to 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_funct}}{must be an integer larger than 1}
//'   \item{\code{basis_degree}}{each element must be an integer larger than or equal to 1}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of corresponding \code{boundary_knots}}
//'   \item{\code{X}}{must have n_funct number of rows (if specified)}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha_nu}}{must be positive}
//'   \item{\code{beta_nu}}{must be positive}
//'   \item{\code{alpha_eta}}{must be positive}
//'   \item{\code{beta_eta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "HDSim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' ## Run function
//' x <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                basis_degree, n_eigen, boundary_knots,
//'                                internal_knots)
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "HDSim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' X <- matrix(rnorm(20, 0 , 1), nrow = 20, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                   basis_degree, n_eigen, boundary_knots,
//'                                   internal_knots, X = X)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List BHDFMMM_Nu_Z_multiple_try(const int tot_mcmc_iters,
                                     const int n_try,
                                     const int K,
                                     const arma::field<arma::vec> Y,
                                     const arma::field<arma::mat> time,
                                     const int n_funct,
                                     const arma::vec basis_degree,
                                     const int n_eigen,
                                     const arma::mat boundary_knots,
                                     const arma::field<arma::vec> internal_knots,
                                     Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                                     const double b = 10,
                                     const double nu_1 = 3,
                                     const double alpha1l = 1,
                                     const double alpha2l= 2,
                                     const double beta1l = 1,
                                     const double beta2l = 1,
                                     const double a_Z_PM = 10000,
                                     const double a_pi_PM = 1000,
                                     const double var_alpha3 = 0.05,
                                     const double var_epsilon1 = 1,
                                     const double var_epsilon2 = 1,
                                     const double alpha_nu = 10,
                                     const double beta_nu = 1,
                                     const double alpha_eta = 10,
                                     const double beta_eta = 1,
                                     const double alpha_0 = 1,
                                     const double beta_0 = 1){
  Rcpp::List BestChain;

  if(X.isNull()){
    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree.n_elem != time(0,0).n_cols){
      Rcpp::stop("number of elemnts in 'basis_degree' does not match number of columns in time matrix");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
    }

    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int j = 0; j < boundary_knots.n_rows; j++){
      for(int i = 0; i < internal_knots(j,0).n_elem; i++){
        if(boundary_knots(j,0) >= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
        }
        if(boundary_knots(j,1) <= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
        }
      }
    }

    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }
    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time, n_funct, basis_degree,
                                                            boundary_knots, internal_knots);

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BHDFMMM_Nu_Z(Y, time, n_funct, K, basis_degree, n_eigen,
                                              boundary_knots, internal_knots,
                                              tot_mcmc_iters, c1, b, alpha1l,
                                              alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                              var_alpha3, var_epsilon1, var_epsilon2,
                                              alpha_nu, beta_nu, alpha_0, beta_0);
    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BHDFMMM_Nu_Z(Y, time, n_funct, K, basis_degree, n_eigen,
                                                boundary_knots, internal_knots,
                                                tot_mcmc_iters, c1, b, alpha1l,
                                                alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                var_alpha3, var_epsilon1, var_epsilon2,
                                                alpha_nu, beta_nu, alpha_0, beta_0);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("B", B_obs),
                                    Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("pi", mod1["pi"]),
                                    Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(X1.n_rows != n_funct){
      Rcpp::stop("'X' must be have 'n_funct' number of rows");
    }
    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree.n_elem != time(0,0).n_cols){
      Rcpp::stop("number of elemnts in 'basis_degree' does not match number of columns in time matrix");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
    }

    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int j = 0; j < boundary_knots.n_rows; j++){
      for(int i = 0; i < internal_knots(j,0).n_elem; i++){
        if(boundary_knots(j,0) >= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
        }
        if(boundary_knots(j,1) <= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
        }
      }
    }

    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha_eta' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta_eta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time, n_funct, basis_degree,
                                                            boundary_knots, internal_knots);

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BHDFMMM_Nu_Z_Cov_Adj(Y, time, X1, n_funct, K, basis_degree,
                                                      n_eigen, boundary_knots, internal_knots,
                                                      tot_mcmc_iters, c1, b, alpha1l,
                                                      alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                      var_alpha3, var_epsilon1, var_epsilon2,
                                                      alpha_nu, beta_nu, alpha_eta, beta_eta,
                                                      alpha_0, beta_0);
    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi =BayesFMMM::BHDFMMM_Nu_Z_Cov_Adj(Y, time, X1, n_funct, K, basis_degree,
                                                       n_eigen, boundary_knots, internal_knots,
                                                       tot_mcmc_iters, c1, b, alpha1l,
                                                       alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                       var_alpha3, var_epsilon1, var_epsilon2,
                                                       alpha_nu, beta_nu, alpha_eta, beta_eta,
                                                       alpha_0, beta_0);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("B", B_obs),
                                    Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("eta", mod1["eta"]),
                                    Rcpp::Named("pi", mod1["pi"]),
                                    Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                    Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }


  return BestChain;
}

//' Find initial starting points for the covariance structure for high dimensional functional data (Domain dimension > 1)
//'
//' This function is meant to be used after using \code{BHDFMMM_NU_Z_multiple_try}.
//' This function samples from the rest of the model parameters given a fixed value of
//' nu and Z. The fixed value of nu and Z are found by using the best markov chain
//' found in \code{BHDFMMM_NU_Z_multiple_try}. Once this function is ran, the results
//' can be used in \code{BHDFMMM_warm_start}. To specify a covariate
//' adjusted model, please provide the design matrix X.
//'
//' @name BHDFMMM_Theta_est
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param n_try Int containing how many different chains are tried
//' @param K Int containing the number of clusters
//' @param Y List of vectors containing the observed values (flattened)
//' @param time List of matrices that contain the observed time points (each column is a dimension)
//' @param n_funct Int containing the number of functions
//' @param basis_degree Vector containing the desired basis degree for each dimension
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Matrix containing the boundary knots for each dimension (each row is a dimension)
//' @param internal_knots List of vectors containing the internal knots for each dimension
//' @param multiple_try List containing results from \code{BHDFMMM_Nu_Z_multiple_try}
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param c Vector containing hyperparmeter for sampling from pi (If left NULL, the one vector will be used)
//' @param b double containing hyperparamete for sampling from alpha_3
//' @param nu_1 double containing hyperparameter for sampling from gamma
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @param covariance_adj Boolean containing whether or not covariates should affect the covariance
//' @returns a List containing:
//' \describe{
//'   \item{\code{B}}{The basis functions evaluated at the observed time points}
//'   \item{\code{Z}}{estimates of Z}
//'   \item{\code{nu}}{estimates of nu}
//'   \item{\code{eta}}{estimated value from \code{BHDFMMM_Nu_Z_multiple_try} (if covariate adjusted)}
//'   \item{\code{chi}}{chi samples from MCMC chain}
//'   \item{\code{A}}{A samples from MCMC chain}
//'   \item{\code{A_xi}}{A_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{delta}}{delta samples from MCMC chain}
//'   \item{\code{delta_xi}}{delta_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{gamma_xi}}{gamma_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{sigma_sq}}{sigma_sq samples from MCMC chain}
//'   \item{\code{tau}}{tau samples from MCMC chain}
//'   \item{\code{tau_eta}}{tau_eta samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{gamma}}{gamma samples from the MCMC chain}
//'   \item{\code{Phi}}{Phi samples from MCMC chain}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{n_try}}{must be an integer larger than or equal to 1}
//'   \item{\code{burnin_prop}}{must be between 0 and 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_funct}}{must be an integer larger than 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{X}}{must have n_funct number of rows (if specified)}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{nu_1}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha_nu}}{must be positive}
//'   \item{\code{beta_nu}}{must be positive}
//'   \item{\code{alpha_eta}}{must be positive}
//'   \item{\code{beta_eta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "HDSim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' ## Run function
//' x <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                basis_degree, n_eigen, boundary_knots,
//'                                internal_knots)
//'
//' ## Run function
//' est2 <- BHDFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                           basis_degree, n_eigen, boundary_knots,
//'                           internal_knots, est1)
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "HDSim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' X <- matrix(rnorm(20, 0 , 1), nrow = 20, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                   basis_degree, n_eigen, boundary_knots,
//'                                   internal_knots, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                         basis_degree, n_eigen, boundary_knots,
//'                         internal_knots, est1, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "HDSim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' X <- matrix(rnorm(20, 0 , 1), nrow = 20, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                   basis_degree, n_eigen, boundary_knots,
//'                                   internal_knots, X = X)
//'
//' ## Run function
//' est2 <- BHDFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                           basis_degree, n_eigen, boundary_knots,
//'                           internal_knots, est1, X = X, covariance_adj = T)
//' @export
// [[Rcpp::export]]
Rcpp::List BHDFMMM_Theta_est(const int tot_mcmc_iters,
                             const int n_try,
                             const int K,
                             const arma::field<arma::vec> Y,
                             const arma::field<arma::mat> time,
                             const int n_funct,
                             const arma::vec basis_degree,
                             const int n_eigen,
                             const arma::mat boundary_knots,
                             const arma::field<arma::vec> internal_knots,
                             const Rcpp::List multiple_try,
                             Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                             const double burnin_prop = 0.8,
                             Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                             const double b = 10,
                             const double nu_1 = 3,
                             const double alpha1l = 2,
                             const double alpha2l = 3,
                             const double beta1l = 2,
                             const double beta2l = 2,
                             const double a_Z_PM = 10000,
                             const double a_pi_PM = 1000,
                             const double var_alpha3 = 0.05,
                             const double var_epsilon1 = 1,
                             const double var_epsilon2 = 1,
                             const double alpha_nu = 10,
                             const double beta_nu = 1,
                             const double alpha_eta = 10,
                             const double beta_eta = 1,
                             const double alpha_0 = 1,
                             const double beta_0 = 1,
                             const bool covariance_adj = false){
  Rcpp::List BestChain;

  if(X.isNull()){
    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree.n_elem != time(0,0).n_cols){
      Rcpp::stop("number of elemnts in 'basis_degree' does not match number of columns in time matrix");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int j = 0; j < boundary_knots.n_rows; j++){
      for(int i = 0; i < internal_knots(j,0).n_elem; i++){
        if(boundary_knots(j,0) >= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
        }
        if(boundary_knots(j,1) <= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
        }
      }
    }

    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time, n_funct, basis_degree,
                                                            boundary_knots, internal_knots);

    int n_nu = nu_samp.n_slices;
    arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BHDFMMM_Theta(Y, time, n_funct, K, basis_degree, n_eigen,
                                               boundary_knots, internal_knots, tot_mcmc_iters,
                                               c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                               beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                               var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                               alpha_0, beta_0, Z_est, nu_est);
    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BHDFMMM_Theta(Y, time, n_funct, K, basis_degree, n_eigen,
                                                 boundary_knots, internal_knots, tot_mcmc_iters,
                                                 c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                                 beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                                 var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                                 alpha_0, beta_0, Z_est, nu_est);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("B", B_obs),
                                    Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("chi", mod1["chi"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("gamma", mod1["gamma"]),
                                    Rcpp::Named("Phi", mod1["Phi"]),
                                    Rcpp::Named("Nu_est", nu_est),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::field<arma::cube> eta_samp = multiple_try["eta"];

    if(eta_samp(0,0).n_rows != nu_samp.n_cols){
      Rcpp::stop("The specified 'Eta' is of the wrong dimension");
    }
    if(X1.n_rows != n_funct){
      Rcpp::stop("'X' must be have 'n_funct' number of rows");
    }
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree.n_elem != time(0,0).n_cols){
      Rcpp::stop("number of elemnts in 'basis_degree' does not match number of columns in time matrix");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int j = 0; j < boundary_knots.n_rows; j++){
      for(int i = 0; i < internal_knots(j,0).n_elem; i++){
        if(boundary_knots(j,0) >= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
        }
        if(boundary_knots(j,1) <= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
        }
      }
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha_eta' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta_eta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time, n_funct, basis_degree,
                                                            boundary_knots, internal_knots);
    int n_nu = nu_samp.n_slices;
    arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::cube eta_est = arma::zeros(eta_samp(0,0).n_rows, eta_samp(0,0).n_cols, eta_samp(0,0).n_slices);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
        for(int d = 0; d < X1.n_cols; d++){
          for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
            ph_eta(l - std::round(n_nu * burnin_prop)) = eta_samp(l,0)(j,d,i);
          }
          eta_est(j,d,i) = arma::median(ph_eta);
        }
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      for(int j = 0; j < Z_est.n_cols; j++){
        Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BHDFMMM_Theta_Cov_Adj(Y, time, X1, n_funct, K, basis_degree, n_eigen,
                                                       boundary_knots, internal_knots, tot_mcmc_iters,
                                                       c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                                       beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                                       var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                                       alpha_eta, beta_eta, alpha_0, beta_0, Z_est, nu_est,
                                                       eta_est, covariance_adj);

    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BHDFMMM_Theta_Cov_Adj(Y, time, X1, n_funct, K, basis_degree, n_eigen,
                                                         boundary_knots, internal_knots, tot_mcmc_iters,
                                                         c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                                         beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                                         var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                                         alpha_eta, beta_eta, alpha_0, beta_0, Z_est, nu_est,
                                                         eta_est, covariance_adj);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain = Rcpp::List::create(Rcpp::Named("B", B_obs),
                                   Rcpp::Named("Z", mod1["Z"]),
                                   Rcpp::Named("nu", mod1["nu"]),
                                   Rcpp::Named("eta", mod1["eta"]),
                                   Rcpp::Named("chi", mod1["chi"]),
                                   Rcpp::Named("A", mod1["A"]),
                                   Rcpp::Named("A_xi", mod1["A_xi"]),
                                   Rcpp::Named("delta", mod1["delta"]),
                                   Rcpp::Named("delta_xi", mod1["delta_xi"]),
                                   Rcpp::Named("gamma_xi", mod1["gamma_xi"]),
                                   Rcpp::Named("sigma_sq", mod1["sigma"]),
                                   Rcpp::Named("tau", mod1["tau"]),
                                   Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                   Rcpp::Named("gamma", mod1["gamma"]),
                                   Rcpp::Named("Phi", mod1["Phi"]),
                                   Rcpp::Named("xi", mod1["xi"]),
                                   Rcpp::Named("loglik", mod1["loglik"]));

  }


  return BestChain;
}


//' Performs MCMC for high dimensional functional model given an informed set of starting points
//'
//' This function is meant to be used after using \code{BHDFMMM_Nu_Z_multiple_try}
//' and \code{BHDFMMM_Theta_est}. This function will use the outputs of these two
//' functions to start the MCMC chain in a good location. Since the posterior distribution
//' can often be multimodal, it is important to have a good starting position.
//' To help move across modes, this function allows users to use tempered transitions
//' every \code{n_temp_trans} iterations. By using a mixture of tempered transitions
//' and un-tempered transitions, we can allow the chain to explore multiple modes without
//' while keeping sampling relatively computationally efficient. To save on RAM usage, we
//' allow users to specify how many samples are kept in memory using \code{r_stored_iters}.
//' If \code{r_stored_iters} is less than \code{tot_mcmc_iters}, then a thinned version
//' of the chain is stored in the user specified directory (\code{dir}). The samples from each
//' parameter can be viewed using the following functions: \code{ReadFieldCube},
//' \code{ReadFieldMat}, \code{ReadFieldVec}, \code{ReadCube}, \code{ReadMat},
//' \code{ReadVec}. To specify a covariate
//' adjusted model, please provide the design matrix X.
//'
//' @name BHDFMMM_warm_start
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param K Int containing the number of clusters
//' @param Y List of vectors containing the observed values
//' @param time List of matrices that contain the observed time points (each column is a dimension)
//' @param n_funct Int containing the number of functions
//' @param basis_degree Vector containing the desired basis degree for each dimension
//' @param n_eigen Int containing the number of eigenfunctions
//' @param boundary_knots Matrix containing the boundary knots for each dimension (each row is a dimension)
//' @param internal_knots List of vectors containing the internal knots for each dimension
//' @param multiple_try List containing results from \code{BFMMM_Nu_Z_multiple_try}
//' @param theta_est List containing results from \code{BFMMM_Theta_est}
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param dir String containing directory where the MCMC files should be saved (if NULL, then no files will be saved)
//' @param thinning_num Int containing how often we should save MCMC iterations
//' @param beta_N_t Double containing the maximum weight for tempered transitions
//' @param N_t Int containing total number of tempered transitions
//' @param n_temp_trans Int containing how often tempered transitions are performed (if 0, then no tempered transitions are performed)
//' @param r_stored_iters Int containing how many MCMC iterations are stored in RAM (if 0, then all MCMC iterations are stored in RAM)
//' @param c Vector containing hyperparmeter for sampling from pi (If left NULL, the one vector will be used)
//' @param b double containing hyperparamete for sampling from alpha_3
//' @param nu_1 double containing hyperparameter for sampling from gamma
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @param covariance_adj Boolean containing whether or not the covariance structure should depend on the covariates
//'
//' @returns a List containing:
//' \describe{
//'   \item{\code{B}}{The basis functions evaluated at the observed time points}
//'   \item{\code{nu}}{Nu samples from the MCMC chain}
//'   \item{\code{chi}}{chi samples from the MCMC chain}
//'   \item{\code{pi}}{pi samples from the MCMC chain}
//'   \item{\code{alpha_3}}{alpha_3 samples from the MCMC chain}
//'   \item{\code{A}}{A samples from MCMC chain}
//'   \item{\code{delta}}{delta samples from the MCMC chain}
//'   \item{\code{sigma_sq}}{sigma_sq samples from the MCMC chain}
//'   \item{\code{tau}}{tau samples from the MCMC chain}
//'   \item{\code{tau_eta}}{tau_eta samples from the MCMC chain (if covariate adjusted)}
//'   \item{\code{eta}}{eta samples from the MCMC chain (if covariate adjusted)}
//'   \item{\code{xi}}{xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{delta_xi}}{delta_xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{gamma_xi}}{gamma_xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{A_xi}}{A_xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{gamma}}{gamma samples from the MCMC chain}
//'   \item{\code{Phi}}{Phi samples from the MCMC chain}
//'   \item{\code{Z}}{Z samples from the MCMC chain}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{X}}{ must have n_funct number of rows}
//'   \item{\code{burnin_prop}}{must be between 0 and 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_funct}}{must be an integer larger than 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{dir}}{must be specified if \code{r_stored_iters} <= \code{tot_mcmc_iters} (other than if \code{r_stored_iters} = 0)}
//'   \item{\code{n_thinning}}{must be a positive integer}
//'   \item{\code{beta_N_t}}{must be between 1 and 0}
//'   \item{\code{N_t}}{must be a positive integer}
//'   \item{\code{n_temp_trans}}{must be a non-negative integer}
//'   \item{\code{r_stored_iters}}{must be a non-negative integer}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{nu_1}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha_nu}}{must be positive}
//'   \item{\code{beta_nu}}{must be positive}
//'   \item{\code{alpha_eta}}{must be positive}
//'   \item{\code{beta_eta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//'@examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "HDSim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' ## Run function
//' x <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                basis_degree, n_eigen, boundary_knots,
//'                                internal_knots)
//'
//' ## Run function
//' est2 <- BHDFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                           basis_degree, n_eigen, boundary_knots,
//'                           internal_knots, est1)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BHDFMMM_warm_start(tot_mcmc_iters, K, Y, time, n_funct,
//'                                  basis_degree, n_eigen, boundary_knots,
//'                                  internal_knots, est1, est2)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "HDSim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' X <- matrix(rnorm(20, 0 , 1), nrow = 20, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                   basis_degree, n_eigen, boundary_knots,
//'                                   internal_knots, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                         basis_degree, n_eigen, boundary_knots,
//'                         internal_knots, est1, X = X)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BHDFMMM_warm_start(tot_mcmc_iters, K, Y, time, n_funct,
//'                                  basis_degree, n_eigen, boundary_knots,
//'                                  internal_knots, est1, est2, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "HDSim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' X <- matrix(rnorm(20, 0 , 1), nrow = 20, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                                   basis_degree, n_eigen, boundary_knots,
//'                                   internal_knots, X = X)
//'
//' ## Run function
//' est2 <- BHDFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y, time, n_funct,
//'                           basis_degree, n_eigen, boundary_knots,
//'                           internal_knots, est1, X = X, covariance_adj = T)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BHDFMMM_warm_start(tot_mcmc_iters, K, Y, time, n_funct,
//'                                  basis_degree, n_eigen, boundary_knots,
//'                                  internal_knots, est1, est2, X = X, covariance_adj = T)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List BHDFMMM_warm_start(const int tot_mcmc_iters,
                              const int K,
                              const arma::field<arma::vec> Y,
                              const arma::field<arma::mat> time,
                              const int n_funct,
                              const arma::vec basis_degree,
                              const int n_eigen,
                              const arma::mat boundary_knots,
                              const arma::field<arma::vec> internal_knots,
                              const Rcpp::List multiple_try,
                              const Rcpp::List theta_est,
                              Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                              const double burnin_prop = 0.8,
                              Rcpp::Nullable<Rcpp::CharacterVector> dir = R_NilValue,
                              const double thinning_num = 1,
                              const double beta_N_t = 1,
                              int N_t = 1,
                              int n_temp_trans = 0,
                              int r_stored_iters = 0,
                              Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                              const double b = 10,
                              const double nu_1 = 3,
                              const double alpha1l = 2,
                              const double alpha2l = 3,
                              const double beta1l = 2,
                              const double beta2l = 2,
                              const double a_Z_PM = 10000,
                              const double a_pi_PM = 1000,
                              const double var_alpha3 = 0.05,
                              const double var_epsilon1 = 1,
                              const double var_epsilon2 = 1,
                              const double alpha_nu = 10,
                              const double beta_nu = 1,
                              const double alpha_eta = 10,
                              const double beta_eta = 1,
                              const double alpha_0 = 1,
                              const double beta_0 = 1,
                              const bool covariance_adj = false){
  Rcpp::List mod2;

  if(X.isNull()){
    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::mat pi_samp = multiple_try["pi"];
    arma::vec alpha_3_samp = multiple_try["alpha_3"];
    arma::cube delta_samp = theta_est["delta"];
    arma::field<arma::cube> gamma_samp = theta_est["gamma"];
    arma::field<arma::cube> Phi_samp = theta_est["Phi"];
    arma::cube A_samp = theta_est["A"];
    arma::mat tau_samp = multiple_try["tau"];
    arma::vec sigma_samp = theta_est["sigma_sq"];
    arma::cube chi_samp = theta_est["chi"];

    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree.n_elem != time(0,0).n_cols){
      Rcpp::stop("number of elemnts in 'basis_degree' does not match number of columns in time matrix");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int j = 0; j < boundary_knots.n_rows; j++){
      for(int i = 0; i < internal_knots(j,0).n_elem; i++){
        if(boundary_knots(j,0) >= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
        }
        if(boundary_knots(j,1) <= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
        }
      }
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }
    if(thinning_num <= 0){
      Rcpp::stop("'thinning_num' must be a positive integer");
    }
    if(beta_N_t <= 0){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(beta_N_t > 1){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(N_t < 1){
      Rcpp::stop("'N_t' must be a positive integer");
    }
    if(r_stored_iters < 0){
      Rcpp::stop("'r_stored_iters' must be a non-negative integer");
    }
    if(n_temp_trans < 0){
      Rcpp::stop("'n_temp_trans' must be a non-negative integer");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()){
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    // if r_stored_iters is default, do not save anything
    std::string dir1 = "";
    if(r_stored_iters == 0){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // check if directory is specified
    if(dir.isNotNull()){
      Rcpp::CharacterVector s(dir);
      dir1 = std::string(s[0]);

      // save entire chain at last iteration
      if(r_stored_iters == 0){
        r_stored_iters = tot_mcmc_iters;
      }
    }

    // Check if there is a place to store files if r_stored_iters < tot_mcmc_iters
    if(dir.isNull()){
      if(r_stored_iters <= tot_mcmc_iters){
        Rcpp::stop("'r_stored_iters' <= 'tot_mcmc_iters' with no 'dir' specified. Either specify 'dir' or increase 'r_stored_iters'");
      }
    }

    // if n_temp_trans is default set to greater than tot_mcmc_iters
    if(n_temp_trans == 0){
      n_temp_trans = tot_mcmc_iters + 1;
      N_t = 1;
    }

    // save RAM
    if(r_stored_iters > tot_mcmc_iters + 1){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // Start of Algorithm
    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time, n_funct, basis_degree,
                                                            boundary_knots, internal_knots);

    int n_nu = alpha_3_samp.n_elem;

    double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    arma::vec pi_est = arma::zeros(pi_samp.n_rows);
    arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < Z_est.n_cols; i++){
      pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
    }

    pi_est = pi_est / arma::accu(pi_est);

    int n_Phi = sigma_samp.n_elem;

    double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
    arma::mat delta_est = arma::zeros(delta_samp.n_rows, delta_samp.n_cols);
    arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < delta_samp.n_cols; j++){
      for(int i = 0; i < delta_samp.n_rows; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,j,l);
        }
        delta_est(i, j) = arma::median(ph_delta);
      }
    }

    arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
    arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
    arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < Phi_est.n_rows; i++){
      for(int j = 0; j < Phi_est.n_cols; j++){
        for(int m = 0; m < Phi_est.n_slices; m++){
          for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
            ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);

            ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
          }
          Phi_est(i,j,m) = arma::median(ph_phi);
          gamma_est(i,j,m) = arma::median(ph_gamma);
        }
      }
    }

    arma::mat A_est = arma::zeros(A_samp.n_rows, A_samp.n_cols);
    arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < A_samp.n_rows; j++){
      for(int i = 0; i < A_samp.n_cols; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(j, i, l);
        }
        A_est(j, i) = arma::median(ph_A);
      }
    }

    arma::vec tau_est = arma::zeros(tau_samp.n_cols);
    arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < tau_est.n_elem; i++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
      }
      tau_est(i) = arma::median(ph_tau);
    }
    arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
    arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < chi_est.n_rows; i++){
      for(int j = 0; j < chi_est.n_cols; j++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
        }
        chi_est(i,j) = arma::median(ph_chi);
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BHDFMMM_MTT_warm_start(Y, time, n_funct, thinning_num, K,
                                                        basis_degree, n_eigen, boundary_knots,
                                                        internal_knots, tot_mcmc_iters,
                                                        r_stored_iters, n_temp_trans,
                                                        c1, b, nu_1, alpha1l, alpha2l,
                                                        beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                        var_alpha3, var_epsilon1,
                                                        var_epsilon2, alpha_nu, beta_nu, alpha_0,
                                                        beta_0, dir1, beta_N_t, N_t,
                                                        Z_est, pi_est, alpha_3_est,
                                                        delta_est, gamma_est, Phi_est, A_est,
                                                        nu_est, tau_est, sigma_est, chi_est);

    mod2 =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                               Rcpp::Named("nu", mod1["nu"]),
                               Rcpp::Named("chi", mod1["chi"]),
                               Rcpp::Named("pi", mod1["pi"]),
                               Rcpp::Named("alpha_3", mod1["alpha_3"]),
                               Rcpp::Named("A", mod1["A"]),
                               Rcpp::Named("delta", mod1["delta"]),
                               Rcpp::Named("sigma_sq", mod1["sigma"]),
                               Rcpp::Named("tau", mod1["tau"]),
                               Rcpp::Named("gamma", mod1["gamma"]),
                               Rcpp::Named("Phi", mod1["Phi"]),
                               Rcpp::Named("Z", mod1["Z"]),
                               Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::mat pi_samp = multiple_try["pi"];
    arma::vec alpha_3_samp = multiple_try["alpha_3"];
    arma::cube delta_samp = theta_est["delta"];
    arma::field<arma::cube> gamma_samp = theta_est["gamma"];
    arma::field<arma::cube> Phi_samp = theta_est["Phi"];
    arma::cube A_samp = theta_est["A"];
    arma::mat tau_samp = multiple_try["tau"];
    arma::vec sigma_samp = theta_est["sigma_sq"];
    arma::cube chi_samp = theta_est["chi"];
    arma::field<arma::cube> eta_samp = multiple_try["eta"];
    arma::cube tau_eta_samp = multiple_try["tau_eta"];

    // generate warnings
    if(X1.n_rows != n_funct){
      Rcpp::stop("'X' must be have 'n_funct' number of rows");
    }
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    if(basis_degree.n_elem != time(0,0).n_cols){
      Rcpp::stop("number of elemnts in 'basis_degree' does not match number of columns in time matrix");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(n_funct <  1){
      Rcpp::stop("'n_funct' must be an integer greater than or equal to 1");
    }
    for(int j = 0; j < boundary_knots.n_rows; j++){
      for(int i = 0; i < internal_knots(j,0).n_elem; i++){
        if(boundary_knots(j,0) >= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
        }
        if(boundary_knots(j,1) <= internal_knots(j,0)(i)){
          Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
        }
      }
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha_eta' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta_eta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }
    if(thinning_num <= 0){
      Rcpp::stop("'thinning_num' must be a positive integer");
    }
    if(beta_N_t <= 0){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(beta_N_t > 1){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(N_t < 1){
      Rcpp::stop("'N_t' must be a positive integer");
    }
    if(r_stored_iters < 0){
      Rcpp::stop("'r_stored_iters' must be a non-negative integer");
    }
    if(n_temp_trans < 0){
      Rcpp::stop("'n_temp_trans' must be a non-negative integer");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()){
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    // if r_stored_iters is default, do not save anything
    std::string dir1 = "";
    if(r_stored_iters == 0){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // check if directory is specified
    if(dir.isNotNull()){
      Rcpp::CharacterVector s(dir);
      dir1 = std::string(s[0]);

      // save entire chain at last iteration
      if(r_stored_iters == 0){
        r_stored_iters = tot_mcmc_iters;
      }
    }

    // Check if there is a place to store files if r_stored_iters < tot_mcmc_iters
    if(dir.isNull()){
      if(r_stored_iters <= tot_mcmc_iters){
        Rcpp::stop("'r_stored_iters' <= 'tot_mcmc_iters' with no 'dir' specified. Either specify 'dir' or increase 'r_stored_iters'");
      }
    }

    // if n_temp_trans is default set to greater than tot_mcmc_iters
    if(n_temp_trans == 0){
      n_temp_trans = tot_mcmc_iters + 1;
      N_t = 1;
    }

    // save RAM
    if(r_stored_iters > tot_mcmc_iters + 1){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // Start of Algorithm
    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time, n_funct, basis_degree,
                                                            boundary_knots, internal_knots);

    int n_nu = alpha_3_samp.n_elem;

    double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    arma::vec pi_est = arma::zeros(pi_samp.n_rows);
    arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::cube eta_est = arma::zeros(eta_samp(0,0).n_rows, eta_samp(0,0).n_cols, eta_samp(0,0).n_slices);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

    for(int i = 0; i < Z_est.n_cols; i++){
      pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    }

    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
        for(int d = 0; d < X1.n_cols; d++){
          for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
            ph_eta(l - std::round(n_nu * burnin_prop)) = eta_samp(l,0)(j,d,i);
          }
          eta_est(j,d,i) = arma::median(ph_eta);
        }
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
    }

    pi_est = pi_est / arma::accu(pi_est);

    int n_Phi = sigma_samp.n_elem;

    double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
    arma::mat delta_est = arma::zeros(delta_samp.n_rows, delta_samp.n_cols);
    arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < delta_samp.n_cols; j++){
      for(int i = 0; i < delta_samp.n_rows; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,j,l);
        }
        delta_est(i, j) = arma::median(ph_delta);
      }
    }

    arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
    arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
    arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < Phi_est.n_rows; i++){
      for(int j = 0; j < Phi_est.n_cols; j++){
        for(int m = 0; m < Phi_est.n_slices; m++){
          for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
            ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);

            ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
          }
          Phi_est(i,j,m) = arma::median(ph_phi);
          gamma_est(i,j,m) = arma::median(ph_gamma);
        }
      }
    }

    arma::mat A_est = arma::zeros(A_samp.n_rows, A_samp.n_cols);
    arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < A_samp.n_rows; j++){
      for(int i = 0; i < A_samp.n_cols; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(j, i, l);
        }
        A_est(j, i) = arma::median(ph_A);
      }
    }

    arma::vec tau_est = arma::zeros(tau_samp.n_cols);
    arma::mat tau_eta_est = arma::zeros(tau_eta_samp.n_rows, tau_eta_samp.n_cols);
    arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_tau_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < tau_est.n_elem; i++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
      }
      tau_est(i) = arma::median(ph_tau);
      for(int j = 0; j < tau_eta_est.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_tau_eta(l - std::round(n_nu * burnin_prop)) = tau_eta_samp(i,j,l);
        }
        tau_eta_est(i,j) = arma::median(ph_tau_eta);
      }
    }
    arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
    arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < chi_est.n_rows; i++){
      for(int j = 0; j < chi_est.n_cols; j++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
        }
        chi_est(i,j) = arma::median(ph_chi);
      }
    }

    Rcpp::List mod2;

    if(covariance_adj == false){
      // start MCMC sampling
      Rcpp::List mod1 = BayesFMMM::BHDFMMM_MTT_warm_start_MeanAdj(Y, time, X1, n_funct, thinning_num, K,
                                                                  basis_degree, n_eigen, boundary_knots,
                                                                  internal_knots, tot_mcmc_iters,
                                                                  r_stored_iters, n_temp_trans,
                                                                  c1, b, nu_1, alpha1l, alpha2l,
                                                                  beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                                  var_alpha3, var_epsilon1,
                                                                  var_epsilon2, alpha_nu, beta_nu,
                                                                  alpha_eta, beta_eta, alpha_0,
                                                                  beta_0, dir1, beta_N_t, N_t,
                                                                  Z_est, pi_est, alpha_3_est,
                                                                  delta_est, gamma_est, Phi_est, A_est,
                                                                  nu_est, eta_est, tau_est, tau_eta_est,
                                                                  sigma_est, chi_est);

      Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                            Rcpp::Named("nu", mod1["nu"]),
                                            Rcpp::Named("chi", mod1["chi"]),
                                            Rcpp::Named("pi", mod1["pi"]),
                                            Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                            Rcpp::Named("A", mod1["A"]),
                                            Rcpp::Named("delta", mod1["delta"]),
                                            Rcpp::Named("sigma_sq", mod1["sigma"]),
                                            Rcpp::Named("tau", mod1["tau"]),
                                            Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                            Rcpp::Named("eta", mod1["eta"]),
                                            Rcpp::Named("gamma", mod1["gamma"]),
                                            Rcpp::Named("Phi", mod1["Phi"]),
                                            Rcpp::Named("Z", mod1["Z"]),
                                            Rcpp::Named("loglik", mod1["loglik"]));
    }else{
      arma::field<arma::cube> delta_xi_samp = theta_est["delta_xi"];
      arma::field<arma::cube> A_xi_samp = theta_est["A_xi"];
      arma::field<arma::cube> xi_samp1 = theta_est["xi"];
      arma::field<arma::cube> xi_samp(n_nu, K);
      for(int k = 0; k < K; k++){
        for(int n = 0; n < n_nu; n++){
          xi_samp(n,k) = xi_samp1(((k * n_nu) + n), 0);
        }
      }
      arma::field<arma::cube> gamma_xi_samp1 = theta_est["gamma_xi"];
      arma::field<arma::cube> gamma_xi_samp(n_nu, K);
      for(int k = 0; k < K; k++){
        for(int n = 0; n < n_nu; n++){
          gamma_xi_samp(n,k) = gamma_xi_samp1(((k * n_nu) + n), 0);
        }
      }

      arma::cube delta_xi_est = arma::zeros(delta_xi_samp(0,0).n_rows,
                                            delta_xi_samp(0,0).n_cols,
                                            delta_xi_samp(0,0).n_slices);
      arma::cube A_xi_est = arma::zeros(A_xi_samp(0,0).n_rows,
                                        A_xi_samp(0,0).n_cols,
                                        A_xi_samp(0,0).n_slices);

      arma::field<arma::cube> xi_est(1,K);
      arma::field<arma::cube> gamma_xi_est(1,K);
      for(int j = 0; j < K; j++){
        xi_est(0,j) = arma::zeros(xi_samp(0,j).n_rows, xi_samp(0,j).n_cols,
               xi_samp(0,j).n_slices);
        gamma_xi_est(0,j) = arma::zeros(gamma_xi_samp(0,j).n_rows,
                     gamma_xi_samp(0,j).n_cols, gamma_xi_samp(0,j).n_slices);
      }

      arma::vec delta_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec A_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec gamma_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

      for(int j = 0; j < K; j++){
        for(int d = 0; d < X1.n_cols; d++){
          for(int i = 0; i < 2; i++){
            for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
              A_xi_ph(l - std::round(n_nu * burnin_prop)) = A_xi_samp(l,0)(j, i, d);
            }
            A_xi_est(j, i, d) = arma::median(A_xi_ph);
          }
          for(int m = 0; m < n_eigen; m++){
            for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
              delta_xi_ph(l - std::round(n_nu * burnin_prop)) = delta_xi_samp(l,0)(j, m, d);
            }
            delta_xi_est(j, m, d) = arma::median(delta_xi_ph);
            for(int p = 0; p < xi_samp(0,j).n_rows; p++){
              for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
                xi_ph(l - std::round(n_nu * burnin_prop)) = xi_samp(l,j)(p, d, m);
                gamma_xi_ph(l - std::round(n_nu * burnin_prop)) = gamma_xi_samp(l,j)(p, d, m);
              }
              xi_est(0,j)(p, d, m) = arma::median(xi_ph);
              gamma_xi_est(0,j)(p, d, m) = arma::median(gamma_xi_ph);
            }
          }
        }
      }
      // start MCMC sampling
      Rcpp::List mod1 = BayesFMMM::BHDFMMM_MTT_warm_start_Mean_CovAdj(Y, time, X1, n_funct, thinning_num, K,
                                                                      basis_degree, n_eigen, boundary_knots,
                                                                      internal_knots, tot_mcmc_iters,
                                                                      r_stored_iters, n_temp_trans,
                                                                      c1, b, nu_1, alpha1l, alpha2l,
                                                                      beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                                      var_alpha3, var_epsilon1,
                                                                      var_epsilon2, alpha_nu, beta_nu,
                                                                      alpha_eta, beta_eta, alpha_0,
                                                                      beta_0, dir1, beta_N_t, N_t,
                                                                      Z_est, pi_est, alpha_3_est,
                                                                      delta_est, delta_xi_est, gamma_est,
                                                                      gamma_xi_est, Phi_est, xi_est, A_est,
                                                                      A_xi_est, nu_est, eta_est, tau_est,
                                                                      tau_eta_est, sigma_est, chi_est);

      mod2 =  Rcpp::List::create(Rcpp::Named("B_obs", B_obs),
                                 Rcpp::Named("nu", mod1["nu"]),
                                 Rcpp::Named("chi", mod1["chi"]),
                                 Rcpp::Named("pi", mod1["pi"]),
                                 Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                 Rcpp::Named("A", mod1["A"]),
                                 Rcpp::Named("delta", mod1["delta"]),
                                 Rcpp::Named("sigma_sq", mod1["sigma"]),
                                 Rcpp::Named("tau", mod1["tau"]),
                                 Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                 Rcpp::Named("xi", mod1["xi"]),
                                 Rcpp::Named("delta_xi", mod1["delta_xi"]),
                                 Rcpp::Named("gamma_xi", mod1["gamma_xi"]),
                                 Rcpp::Named("A_xi", mod1["A_xi"]),
                                 Rcpp::Named("eta", mod1["eta"]),
                                 Rcpp::Named("gamma", mod1["gamma"]),
                                 Rcpp::Named("Phi", mod1["Phi"]),
                                 Rcpp::Named("Z", mod1["Z"]),
                                 Rcpp::Named("loglik", mod1["loglik"]));
    }
  }


  return mod2;
}

//' Find initial starting position for mean and allocation structure for multivariate data
//'
//' Function for finding a good initial starting point for nu parameters, Z, and eta parameters (if covariate adjusted)
//' for multivariate data, with option for tempered transitions.This
//' function tries running multiple different MCMC chains to find the optimal
//' starting position. This function will return the chain that has the highest
//' log-likelihood average in the last 100 MCMC iterations. To specify a covariate
//' adjusted model, please provide the design matrix X.
//'
//' @name BMVMMM_Nu_Z_multiple_try
//' @param tot_mcmc_iters Int containing the number of MCMC iterations per try
//' @param n_try Int containing how many different chains are tried
//' @param K Int containing the number of clusters
//' @param Y Matrix of observed vectors (each row is an observation)
//' @param n_eigen Int containing the number of eigenfunctions
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param c Vector containing hyperparmeters for sampling from pi (If left NULL, the one vector will be used)
//' @param b Double containing hyperparameter for sampling from alpha_3
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameters for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparamete for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @returns a List containing:
//' \describe{
//'   \item{\code{nu}}{Nu samples from the chain with the highest average log-likelihood}
//'   \item{\code{eta}}{Eta samples from the chain with the highest average log-likelihood (if covariate adjusted)}
//'   \item{\code{pi}}{Pi samples from the chain with the highest average log-likelihood}
//'   \item{\code{alpha_3}}{Alpha_3 samples from the chain with the highest average log-likelihood}
//'   \item{\code{A}}{A samples from the chain with the highest average log-likelihood}
//'   \item{\code{delta}}{Delta samples from the chain with the highest average log-likelihood}
//'   \item{\code{sigma_sq}}{sigma_sq samples from the chain with the highest average log-likelihood}
//'   \item{\code{tau}}{Tau samples from the chain with the highest average log-likelihood}
//'   \item{\code{tau_eta}}{Tau_Eta samples from the chain with the highest average log-likelihood (if covariate adjusted)}
//'   \item{\code{Z}}{Z samples from the chain with the highest average log-likelihood}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{n_try}}{must be an integer larger than or equal to 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{X}}{must have the same number of rows as Y (if specified)}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha_nu}}{must be positive}
//'   \item{\code{beta_nu}}{must be positive}
//'   \item{\code{alpha_eta}}{must be positive}
//'   \item{\code{beta_eta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_eigen <- 2
//'
//' ## Run function
//' est1 <- BMVMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, n_eigen)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_eigen <- 2
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Run function
//' est1 <- BMVMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, n_eigen, X = X)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List BMVMMM_Nu_Z_multiple_try(const int tot_mcmc_iters,
                                    const int n_try,
                                    const int K,
                                    const arma::mat Y,
                                    const int n_eigen,
                                    Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                                    Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                                    const double b = 10,
                                    const double alpha1l = 2,
                                    const double alpha2l= 3,
                                    const double beta1l = 1,
                                    const double beta2l = 1,
                                    const double a_Z_PM = 10000,
                                    const double a_pi_PM = 1000,
                                    const double var_alpha3 = 0.05,
                                    const double var_epsilon1 = 1,
                                    const double var_epsilon2 = 1,
                                    const double alpha_nu = 10,
                                    const double beta_nu = 1,
                                    const double alpha_eta = 10,
                                    const double beta_eta = 1,
                                    const double alpha_0 = 1,
                                    const double beta_0 = 1){
  Rcpp::List BestChain;

  if(X.isNull()){
    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }

    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }

    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_Nu_ZMV(Y, K, n_eigen,
                                              tot_mcmc_iters, c1, b, alpha1l,
                                              alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                              var_alpha3, var_epsilon1, var_epsilon2,
                                              alpha_nu, beta_nu, alpha_0, beta_0);
    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BFMMM_Nu_ZMV(Y, K, n_eigen,
                                                tot_mcmc_iters, c1, b, alpha1l,
                                                alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                var_alpha3, var_epsilon1, var_epsilon2,
                                                alpha_nu, beta_nu, alpha_0, beta_0);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("pi", mod1["pi"]),
                                    Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(X1.n_rows != Y.n_rows){
      Rcpp::stop("'X' must be have the same number of rows as Y");
    }

    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }

    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }

    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha_eta' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta_eta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_Nu_ZMV_Cov_Adj(Y, X1, K, n_eigen,
                                                      tot_mcmc_iters, c1, b, alpha1l,
                                                      alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                      var_alpha3, var_epsilon1, var_epsilon2,
                                                      alpha_nu, beta_nu, alpha_eta, beta_eta,
                                                      alpha_0, beta_0);
    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BFMMM_Nu_ZMV_Cov_Adj(Y, X1, K, n_eigen,
                                                        tot_mcmc_iters, c1, b, alpha1l,
                                                        alpha2l, beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                        var_alpha3, var_epsilon1, var_epsilon2,
                                                        alpha_nu, beta_nu, alpha_eta, beta_eta,
                                                        alpha_0, beta_0);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("eta", mod1["eta"]),
                                    Rcpp::Named("pi", mod1["pi"]),
                                    Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                    Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }

  return BestChain;
}

//' Find initial starting points for covariance parameters given mean and allocation structure
//' for multivariate data
//'
//' This function is meant to be used after using \code{BMVMMM_NU_Z_multiple_try}.
//' This function samples from the rest of the model parameters given a fixed value of
//' nu and Z. The fixed value of nu and Z are found by using the best markov chain
//' found in \code{BMVMMM_NU_Z_multiple_try}. Once this function is ran, the results
//' can be used in \code{BMVMMM_warm_start}.To specify a covariate
//' adjusted model, please provide the design matrix X.
//'
//' @name BMVMMM_Theta_est
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param n_try Int containing how many different chains are tried
//' @param K Int containing the number of clusters
//' @param Y Matrix of observed vectors (each row is an observation)
//' @param n_eigen Int containing the number of eigenfunctions
//' @param multiple_try List containing results from \code{BMVMMM_Nu_Z_multiple_try}
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param c Vector containing hyperparmeter for sampling from pi (If left NULL, the one vector will be used)
//' @param b double containing hyperparamete for sampling from alpha_3
//' @param nu_1 double containing hyperparameter for sampling from gamma
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @param covariance_adj Boolean containing whether or not covariates should affect the covariance
//' @returns a List containing:
//' \describe{
//'   \item{\code{Z}}{estimates of Z}
//'   \item{\code{nu}}{estimates of nu}
//'   \item{\code{eta}}{estimated value from \code{BFMMM_Nu_Z_multiple_try} (if covariate adjusted)}
//'   \item{\code{chi}}{chi samples from MCMC chain}
//'   \item{\code{A}}{A samples from MCMC chain}
//'   \item{\code{A_xi}}{A_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{delta}}{delta samples from MCMC chain}
//'   \item{\code{delta_xi}}{delta_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{gamma_xi}}{gamma_xi samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{sigma_sq}}{sigma_sq samples from MCMC chain}
//'   \item{\code{tau}}{tau samples from MCMC chain}
//'   \item{\code{tau_eta}}{tau_eta samples from MCMC chain (if covariate adjusted)}
//'   \item{\code{gamma}}{gamma samples from the MCMC chain}
//'   \item{\code{Phi}}{Phi samples from MCMC chain}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{n_try}}{must be an integer larger than or equal to 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{X}}{must have the same number of rows as Y (if specified)}
//'   \item{\code{burnin_prop}}{must be between 0 and 1}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{nu_1}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha_nu}}{must be positive}
//'   \item{\code{beta_nu}}{must be positive}
//'   \item{\code{alpha_eta}}{must be positive}
//'   \item{\code{beta_eta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_eigen <- 2
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, n_eigen)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y,
//'                         n_eigen, est1)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_eigen <- 2
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, n_eigen, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y,
//'                         n_eigen, est1, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_eigen <- 2
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, n_eigen, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y,
//'                         n_eigen, est1, X = X, covariance_adj = T)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List BMVMMM_Theta_est(const int tot_mcmc_iters,
                            const int n_try,
                            const int K,
                            const arma::mat Y,
                            const int n_eigen,
                            const Rcpp::List multiple_try,
                            Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                            const double burnin_prop = 0.8,
                            Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                            const double b = 10,
                            const double nu_1 = 3,
                            const double alpha1l = 1,
                            const double alpha2l = 2,
                            const double beta1l = 1,
                            const double beta2l = 1,
                            const double a_Z_PM = 10000,
                            const double a_pi_PM = 1000,
                            const double var_alpha3 = 0.05,
                            const double var_epsilon1 = 1,
                            const double var_epsilon2 = 1,
                            const double alpha_nu = 10,
                            const double beta_nu = 1,
                            const double alpha_eta = 10,
                            const double beta_eta = 1,
                            const double alpha_0 = 1,
                            const double beta_0 = 1,
                            const bool covariance_adj = false){
  Rcpp::List BestChain;

  if(X.isNull()){
    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];

    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    int n_nu = nu_samp.n_slices;
    arma::mat Z_est = arma::zeros(Y.n_rows, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      for(int j = 0; j < Z_est.n_cols; j++){
        Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_ThetaMV(Y, K, n_eigen, tot_mcmc_iters,
                                               c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                               beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                               var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                               alpha_0, beta_0, Z_est, nu_est);

    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BFMMM_ThetaMV(Y, K, n_eigen, tot_mcmc_iters,
                                                 c1, b, nu_1, alpha1l, alpha2l, beta1l,
                                                 beta2l, a_Z_PM, a_pi_PM, var_alpha3,
                                                 var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                                 alpha_0, beta_0, Z_est, nu_est);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain =  Rcpp::List::create(Rcpp::Named("Z", mod1["Z"]),
                                    Rcpp::Named("nu", mod1["nu"]),
                                    Rcpp::Named("chi", mod1["chi"]),
                                    Rcpp::Named("A", mod1["A"]),
                                    Rcpp::Named("delta", mod1["delta"]),
                                    Rcpp::Named("sigma_sq", mod1["sigma"]),
                                    Rcpp::Named("tau", mod1["tau"]),
                                    Rcpp::Named("gamma", mod1["gamma"]),
                                    Rcpp::Named("Phi", mod1["Phi"]),
                                    Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::field<arma::cube> eta_samp = multiple_try["eta"];

    if(eta_samp(0,0).n_rows != nu_samp.n_cols){
      Rcpp::stop("The specified 'Eta' is of the wrong dimension");
    }
    if(X1.n_rows != Y.n_rows){
      Rcpp::stop("'X' must be have the same number of rows as Y");
    }
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(n_try <  1){
      Rcpp::stop("'n_try' must be an integer greater than or equal to 1");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha_eta' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta_eta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()) {
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    int n_nu = nu_samp.n_slices;
    arma::mat Z_est = arma::zeros(Y.n_rows, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::cube eta_est = arma::zeros(eta_samp(0,0).n_rows, eta_samp(0,0).n_cols, eta_samp(0,0).n_slices);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
        for(int d = 0; d < X1.n_cols; d++){
          for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
            ph_eta(l - std::round(n_nu * burnin_prop)) = eta_samp(l,0)(j,d,i);
          }
          eta_est(j,d,i) = arma::median(ph_eta);
        }
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      for(int j = 0; j < Z_est.n_cols; j++){
        Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
      }
    }
    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_ThetaMV_Cov_Adj(Y, X1, K, n_eigen,
                                                       tot_mcmc_iters, c1, b, nu_1,
                                                       alpha1l, alpha2l, beta1l, beta2l,
                                                       a_Z_PM, a_pi_PM, var_alpha3,
                                                       var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                                       alpha_eta, beta_eta, alpha_0, beta_0, Z_est, nu_est,
                                                       eta_est, covariance_adj);

    arma::vec ph = mod1["loglik"];
    double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));

    for(int i = 0; i < n_try; i++){
      Rcpp::Rcout << "Try: " << i+1 << " out of " << n_try << "\n";
      Rcpp::List modi = BayesFMMM::BFMMM_ThetaMV_Cov_Adj(Y, X1, K, n_eigen,
                                                         tot_mcmc_iters, c1, b, nu_1,
                                                         alpha1l, alpha2l, beta1l, beta2l,
                                                         a_Z_PM, a_pi_PM, var_alpha3,
                                                         var_epsilon1, var_epsilon2, alpha_nu, beta_nu,
                                                         alpha_eta, beta_eta, alpha_0, beta_0, Z_est, nu_est,
                                                         eta_est, covariance_adj);
      arma::vec ph1 = modi["loglik"];
      if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
        mod1 = modi;
        min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
      }

    }

    BestChain = Rcpp::List::create(Rcpp::Named("Z", mod1["Z"]),
                                   Rcpp::Named("nu", mod1["nu"]),
                                   Rcpp::Named("eta", mod1["eta"]),
                                   Rcpp::Named("chi", mod1["chi"]),
                                   Rcpp::Named("A", mod1["A"]),
                                   Rcpp::Named("A_xi", mod1["A_xi"]),
                                   Rcpp::Named("delta", mod1["delta"]),
                                   Rcpp::Named("delta_xi", mod1["delta_xi"]),
                                   Rcpp::Named("gamma_xi", mod1["gamma_xi"]),
                                   Rcpp::Named("sigma_sq", mod1["sigma"]),
                                   Rcpp::Named("tau", mod1["tau"]),
                                   Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                   Rcpp::Named("gamma", mod1["gamma"]),
                                   Rcpp::Named("Phi", mod1["Phi"]),
                                   Rcpp::Named("xi", mod1["xi"]),
                                   Rcpp::Named("loglik", mod1["loglik"]));
  }

  return BestChain;
}

//' Performs MCMC for multivariate models given an informed set of starting points
//'
//' This function is meant to be used after using \code{BMVMMM_Nu_Z_multiple_try}
//' and \code{BMVMMM_Theta_est}. This function will use the outputs of these two
//' functions to start the MCMC chain in a good location. Since the posterior distribution
//' can often be multimodal, it is important to have a good starting position.
//' To help move across modes, this function allows users to use tempered transitions
//' every \code{n_temp_trans} iterations. By using a mixture of tempered transitions
//' and un-tempered transitions, we can allow the chain to explore multiple modes without
//' while keeping sampling relatively computationally efficient. To save on RAM usage, we
//' allow users to specify how many samples are kept in memory using \code{r_stored_iters}.
//' If \code{r_stored_iters} is less than \code{tot_mcmc_iters}, then a thinned version
//' of the chain is stored in the user specified directory (\code{dir}). The samples from each
//' parameter can be viewed using the following functions: \code{ReadFieldCube},
//' \code{ReadFieldMat}, \code{ReadFieldVec}, \code{ReadCube}, \code{ReadMat},
//' \code{ReadVec}. To specify a covariate adjusted model, please provide the design matrix X.
//'
//' @name BMVMMM_warm_start
//' @param tot_mcmc_iters Int containing the total number of MCMC iterations
//' @param K Int containing the number of clusters
//' @param Y Matrix of observed vectors (each row is an observation)
//' @param n_eigen Int containing the number of eigenfunctions
//' @param multiple_try List containing results from \code{BMVMMM_Nu_Z_multiple_try}
//' @param theta_est List containing results from \code{BMVMMM_Theta_est}
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param burnin_prop Double containing proportion of chain used to estimate the starting point of nu parameters and Z parameters
//' @param dir String containing directory where the MCMC files should be saved (if NULL, then no files will be saved)
//' @param thinning_num Int containing how often we should save MCMC iterations
//' @param beta_N_t Double containing the maximum weight for tempered transitions
//' @param N_t Int containing total number of tempered transitions
//' @param n_temp_trans Int containing how often tempered transitions are performed (if 0, then no tempered transitions are performed)
//' @param r_stored_iters Int containing how many MCMC iterations are stored in RAM (if 0, then all MCMC iterations are stored in RAM)
//' @param c Vector containing hyperparmeter for sampling from pi (If left NULL, the one vector will be used)
//' @param b double containing hyperparamete for sampling from alpha_3
//' @param nu_1 double containing hyperparameter for sampling from gamma
//' @param alpha1l Double containing hyperparameter for sampling from A
//' @param alpha2l Double containing hyperparameter for sampling from A
//' @param beta1l Double containing hyperparameter for sampling from A (scale)
//' @param beta2l Double containing hyperparameter for sampling from A (scale)
//' @param a_Z_PM Double containing hyperparameter of the random walk MH for Z parameter
//' @param a_pi_PM Double containing hyperparameter of the random walk MH for pi parameter
//' @param var_alpha3 Double containing variance parameter of the random walk MH for alpha_3 parameter
//' @param var_epsilon1 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param var_epsilon2 Double containing hyperparameter for sampling from A having to do with variance for Metropolis-Hastings algorithm
//' @param alpha_nu Double containing hyperparameter for sampling from tau_nu
//' @param beta_nu Double containing hyperparameter for sampling from tau_nu (scale)
//' @param alpha_eta Double containing hyperparameter for sampling from tau_eta
//' @param beta_eta Double containing hyperparameter for sampling from tau_eta (scale)
//' @param alpha_0 Double containing hyperparameter for sampling from sigma_sq
//' @param beta_0 Double containing hyperparameter for sampling from sigma_sq (scale)
//' @param covariance_adj Boolean containing whether or not the covariance structure should depend on the covariates
//'
//' @returns a List containing:
//' \describe{
//'   \item{\code{nu}}{Nu samples from the MCMC chain}
//'   \item{\code{chi}}{chi samples from the MCMC chain}
//'   \item{\code{pi}}{pi samples from the MCMC chain}
//'   \item{\code{alpha_3}}{alpha_3 samples from the MCMC chain}
//'   \item{\code{A}}{A samples from MCMC chain}
//'   \item{\code{delta}}{delta samples from the MCMC chain}
//'   \item{\code{sigma_sq}}{sigma_sq samples from the MCMC chain}
//'   \item{\code{tau}}{tau samples from the MCMC chain}
//'   \item{\code{tau_eta}}{tau_eta samples from the MCMC chain}
//'   \item{\code{eta}}{eta samples from the MCMC chain}
//'   \item{\code{xi}}{xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{delta_xi}}{delta_xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{gamma_xi}}{gamma_xi samples from the MCMC chain (if covariance_adj is true)}
//'   \item{\code{gamma}}{gamma samples from the MCMC chain}
//'   \item{\code{Phi}}{Phi samples from the MCMC chain}
//'   \item{\code{Z}}{Z samples from the MCMC chain}
//'   \item{\code{loglik}}{Log-likelihood plot of best performing chain}
//' }
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{tot_mcmc_iters}}{must be an integer larger than or equal to 100}
//'   \item{\code{X}}{must have the same number of rows as Y}
//'   \item{\code{burnin_prop}}{must be between 0 and 1}
//'   \item{\code{K}}{must be an integer larger than or equal to 2}
//'   \item{\code{n_eigen}}{must be greater than or equal to 1}
//'   \item{\code{dir}}{must be specified if \code{r_stored_iters} <= \code{tot_mcmc_iters} (other than if \code{r_stored_iters} = 0)}
//'   \item{\code{n_thinning}}{must be a positive integer}
//'   \item{\code{beta_N_t}}{must be between 1 and 0}
//'   \item{\code{N_t}}{must be a positive integer}
//'   \item{\code{n_temp_trans}}{must be a non-negative integer}
//'   \item{\code{r_stored_iters}}{must be a non-negative integer}
//'   \item{\code{c}}{must be greater than 0 and have K elements}
//'   \item{\code{b}}{must be positive}
//'   \item{\code{nu_1}}{must be positive}
//'   \item{\code{alpha1l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{alpha2l}}{must be positive}
//'   \item{\code{beta1l}}{must be positive}
//'   \item{\code{a_Z_PM}}{must be positive}
//'   \item{\code{a_pi_PM}}{must be positive}
//'   \item{\code{var_alpha3}}{must be positive}
//'   \item{\code{var_epsilon1}}{must be positive}
//'   \item{\code{var_epsilon2}}{must be positive}
//'   \item{\code{alpha}}{must be positive}
//'   \item{\code{beta}}{must be positive}
//'   \item{\code{alpha_0}}{must be positive}
//'   \item{\code{beta_0}}{must be positive}
//' }
//'
//'@examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_eigen <- 2
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, n_eigen)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y,
//'                         n_eigen, est1)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BFMMM_warm_start(tot_mcmc_iters, K, Y, n_eigen, est1, est2)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_eigen <- 2
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, n_eigen, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y,
//'                         n_eigen, est1, X = X)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BFMMM_warm_start(tot_mcmc_iters, K, Y, n_eigen, est1, est2, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Load sample data
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//'
//' ## Set Hyperparameters
//' tot_mcmc_iters <- 150
//' n_try <- 1
//' K <- 2
//' n_eigen <- 2
//'
//' X <- matrix(rnorm(40, 0 , 1), nrow = 40, ncol = 1)
//'
//' ## Get Estimates of Z and nu
//' est1 <- BFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, K, Y, n_eigen, X = X)
//'
//' ## Run function
//' est2 <- BFMMM_Theta_est(tot_mcmc_iters, n_try, K, Y,
//'                         n_eigen, est1, X = X, covariance_adj = T)
//'
//' ## Run MCMC sampler
//' MCMC.chain <- BFMMM_warm_start(tot_mcmc_iters, K, Y, n_eigen, est1, est2, X = X,
//'                                covariance_adj = T)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List BMVMMM_warm_start(const int tot_mcmc_iters,
                             const int K,
                             const arma::mat Y,
                             const int n_eigen,
                             const Rcpp::List multiple_try,
                             const Rcpp::List theta_est,
                             Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                             const double burnin_prop = 0.8,
                             Rcpp::Nullable<Rcpp::CharacterVector> dir = R_NilValue,
                             const double thinning_num = 1,
                             const double beta_N_t = 1,
                             int N_t = 1,
                             int n_temp_trans = 0,
                             int r_stored_iters = 0,
                             Rcpp::Nullable<Rcpp::NumericVector> c  = R_NilValue,
                             const double b = 10,
                             const double nu_1 = 3,
                             const double alpha1l = 1,
                             const double alpha2l = 2,
                             const double beta1l = 1,
                             const double beta2l = 1,
                             const double a_Z_PM = 10000,
                             const double a_pi_PM = 1000,
                             const double var_alpha3 = 0.05,
                             const double var_epsilon1 = 1,
                             const double var_epsilon2 = 1,
                             const double alpha_nu = 10,
                             const double beta_nu = 1,
                             const double alpha_eta = 10,
                             const double beta_eta = 1,
                             const double alpha_0 = 1,
                             const double beta_0 = 1,
                             const bool covariance_adj = false){
  Rcpp::List mod2;

  if(X.isNull()){
    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::mat pi_samp = multiple_try["pi"];
    arma::vec alpha_3_samp = multiple_try["alpha_3"];
    arma::cube delta_samp = theta_est["delta"];
    arma::field<arma::cube> gamma_samp = theta_est["gamma"];
    arma::field<arma::cube> Phi_samp = theta_est["Phi"];
    arma::cube A_samp = theta_est["A"];
    arma::mat tau_samp = multiple_try["tau"];
    arma::vec sigma_samp = theta_est["sigma_sq"];
    arma::cube chi_samp = theta_est["chi"];

    // generate warnings
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }
    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }
    if(thinning_num <= 0){
      Rcpp::stop("'thinning_num' must be a positive integer");
    }
    if(beta_N_t <= 0){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(beta_N_t > 1){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(N_t < 1){
      Rcpp::stop("'N_t' must be a positive integer");
    }
    if(r_stored_iters < 0){
      Rcpp::stop("'r_stored_iters' must be a non-negative integer");
    }
    if(n_temp_trans < 0){
      Rcpp::stop("'n_temp_trans' must be a non-negative integer");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()){
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    // if r_stored_iters is default, do not save anything
    std::string dir1 = "";
    if(r_stored_iters == 0){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // check if directory is specified
    if(dir.isNotNull()){
      Rcpp::CharacterVector s(dir);
      dir1 = std::string(s[0]);

      // save entire chain at last iteration
      if(r_stored_iters == 0){
        r_stored_iters = tot_mcmc_iters;
      }
    }

    // Check if there is a place to store files if r_stored_iters < tot_mcmc_iters
    if(dir.isNull()){
      if(r_stored_iters <= tot_mcmc_iters){
        Rcpp::stop("'r_stored_iters' <= 'tot_mcmc_iters' with no 'dir' specified. Either specify 'dir' or increase 'r_stored_iters'");
      }
    }

    // if n_temp_trans is default set to greater than tot_mcmc_iters
    if(n_temp_trans == 0){
      n_temp_trans = tot_mcmc_iters + 1;
      N_t = 1;
    }

    // save RAM
    if(r_stored_iters > tot_mcmc_iters + 1){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // Start of Algorithm

    int n_nu = alpha_3_samp.n_elem;

    double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    arma::vec pi_est = arma::zeros(pi_samp.n_rows);
    arma::mat Z_est = arma::zeros(Y.n_rows, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < Z_est.n_cols; i++){
      pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
    }

    pi_est = pi_est / arma::accu(pi_est);

    int n_Phi = sigma_samp.n_elem;

    double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
    arma::mat delta_est = arma::zeros(delta_samp.n_rows, delta_samp.n_cols);
    arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < delta_samp.n_cols; j++){
      for(int i = 0; i < delta_samp.n_rows; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,j,l);
        }
        delta_est(i, j) = arma::median(ph_delta);
      }
    }

    arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
    arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
    arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < Phi_est.n_rows; i++){
      for(int j = 0; j < Phi_est.n_cols; j++){
        for(int m = 0; m < Phi_est.n_slices; m++){
          for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
            ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);

            ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
          }
          Phi_est(i,j,m) = arma::median(ph_phi);
          gamma_est(i,j,m) = arma::median(ph_gamma);
        }
      }
    }

    arma::mat A_est = arma::zeros(A_samp.n_rows, A_samp.n_cols);
    arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < A_samp.n_rows; j++){
      for(int i = 0; i < A_samp.n_cols; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(j, i, l);
        }
        A_est(j, i) = arma::median(ph_A);
      }
    }

    arma::vec tau_est = arma::zeros(tau_samp.n_cols);
    arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < tau_est.n_elem; i++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
      }
      tau_est(i) = arma::median(ph_tau);
    }
    arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
    arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < chi_est.n_rows; i++){
      for(int j = 0; j < chi_est.n_cols; j++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
        }
        chi_est(i,j) = arma::median(ph_chi);
      }
    }

    // start MCMC sampling
    Rcpp::List mod1 = BayesFMMM::BFMMM_MTT_warm_startMV(Y, thinning_num, K,
                                                        n_eigen, tot_mcmc_iters,
                                                        r_stored_iters, n_temp_trans,
                                                        c1, b, nu_1, alpha1l, alpha2l,
                                                        beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                        var_alpha3, var_epsilon1,
                                                        var_epsilon2, alpha_nu, beta_nu, alpha_0,
                                                        beta_0, dir1, beta_N_t, N_t,
                                                        Z_est, pi_est, alpha_3_est,
                                                        delta_est, gamma_est, Phi_est, A_est,
                                                        nu_est, tau_est, sigma_est, chi_est);

    mod2 =  Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                               Rcpp::Named("chi", mod1["chi"]),
                               Rcpp::Named("pi", mod1["pi"]),
                               Rcpp::Named("alpha_3", mod1["alpha_3"]),
                               Rcpp::Named("A", mod1["A"]),
                               Rcpp::Named("delta", mod1["delta"]),
                               Rcpp::Named("sigma_sq", mod1["sigma_sq"]),
                               Rcpp::Named("tau", mod1["tau"]),
                               Rcpp::Named("gamma", mod1["gamma"]),
                               Rcpp::Named("Phi", mod1["Phi"]),
                               Rcpp::Named("Z", mod1["Z"]),
                               Rcpp::Named("loglik", mod1["loglik"]));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    arma::cube Z_samp = multiple_try["Z"];
    arma::cube nu_samp = multiple_try["nu"];
    arma::mat pi_samp = multiple_try["pi"];
    arma::vec alpha_3_samp = multiple_try["alpha_3"];
    arma::cube delta_samp = theta_est["delta"];
    arma::field<arma::cube> gamma_samp = theta_est["gamma"];
    arma::field<arma::cube> Phi_samp = theta_est["Phi"];
    arma::cube A_samp = theta_est["A"];
    arma::mat tau_samp = multiple_try["tau"];
    arma::vec sigma_samp = theta_est["sigma_sq"];
    arma::cube chi_samp = theta_est["chi"];
    arma::field<arma::cube> eta_samp = multiple_try["eta"];
    arma::cube tau_eta_samp = multiple_try["tau_eta"];

    if(eta_samp(0,0).n_rows != nu_samp.n_cols){
      Rcpp::stop("The specified 'Eta' is of the wrong dimension");
    }
    if(X1.n_rows != Y.n_rows){
      Rcpp::stop("'X' must be have the same number of rows as Y");
    }
    if(tot_mcmc_iters <  100){
      Rcpp::stop("'tot_mcmc_iters' must be an integer greater than or equal to 100");
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(K <  2){
      Rcpp::stop("'K' must be an integer greater than or equal to 2");
    }
    if(n_eigen <  1){
      Rcpp::stop("'n_eigen' must be an integer greater than or equal to 1");
    }

    if(b <= 0){
      Rcpp::stop("'b' must be positive");
    }
    if(nu_1 <= 0){
      Rcpp::stop("'nu_1' must be positive");
    }
    if(alpha1l <= 0){
      Rcpp::stop("'alpha1l' must be positive");
    }
    if(beta1l <= 0){
      Rcpp::stop("'beta1l' must be positive");
    }
    if(alpha2l <= 0){
      Rcpp::stop("'alpha2l' must be positive");
    }
    if(beta2l <= 0){
      Rcpp::stop("'beta2l' must be positive");
    }
    if(a_Z_PM <= 0){
      Rcpp::stop("'a_Z_PM' must be positive");
    }
    if(a_pi_PM <= 0){
      Rcpp::stop("'a_pi_PM' must be positive");
    }
    if(var_alpha3 <= 0){
      Rcpp::stop("'var_alpha3' must be positive");
    }
    if(var_epsilon1 <= 0){
      Rcpp::stop("'var_epsilon1' must be positive");
    }
    if(var_epsilon2 <= 0){
      Rcpp::stop("'var_epsilon2' must be positive");
    }
    if(alpha_nu <= 0){
      Rcpp::stop("'alpha_nu' must be positive");
    }
    if(beta_nu <= 0){
      Rcpp::stop("'beta_nu' must be positive");
    }
    if(alpha_eta <= 0){
      Rcpp::stop("'alpha_eta' must be positive");
    }
    if(beta_eta <= 0){
      Rcpp::stop("'beta_eta' must be positive");
    }
    if(alpha_0 <= 0){
      Rcpp::stop("'alpha_0' must be positive");
    }
    if(beta_0 <= 0){
      Rcpp::stop("'beta_0' must be positive");
    }
    if(thinning_num <= 0){
      Rcpp::stop("'thinning_num' must be a positive integer");
    }
    if(beta_N_t <= 0){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(beta_N_t > 1){
      Rcpp::stop("'beta_N_t' must be between 0 and 1");
    }
    if(N_t < 1){
      Rcpp::stop("'N_t' must be a positive integer");
    }
    if(r_stored_iters < 0){
      Rcpp::stop("'r_stored_iters' must be a non-negative integer");
    }
    if(n_temp_trans < 0){
      Rcpp::stop("'n_temp_trans' must be a non-negative integer");
    }

    // initialize hyperparameter c
    arma::vec c1 = arma::ones(K) * 10;
    if(c.isNotNull()){
      Rcpp::NumericVector c_(c);
      c1 = Rcpp::as<arma::vec>(c_);
    }

    // generate warning for c
    if(c1.n_elem != K){
      Rcpp::stop("number of elements of the vector 'c' must be equal to K");
    }
    for(int i = 0; i < K; i++){
      if(c1(i) <= 0){
        Rcpp::stop("all elements of 'c' must be positive");
      }
    }

    // if r_stored_iters is default, do not save anything
    std::string dir1 = "";
    if(r_stored_iters == 0){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    // check if directory is specified
    if(dir.isNotNull()){
      Rcpp::CharacterVector s(dir);
      dir1 = std::string(s[0]);

      // save entire chain at last iteration
      if(r_stored_iters == 0){
        r_stored_iters = tot_mcmc_iters;
      }
    }

    // Check if there is a place to store files if r_stored_iters < tot_mcmc_iters
    if(dir.isNull()){
      if(r_stored_iters <= tot_mcmc_iters){
        Rcpp::stop("'r_stored_iters' <= 'tot_mcmc_iters' with no 'dir' specified. Either specify 'dir' or increase 'r_stored_iters'");
      }
    }

    // if n_temp_trans is default set to greater than tot_mcmc_iters
    if(n_temp_trans == 0){
      n_temp_trans = tot_mcmc_iters + 1;
      N_t = 1;
    }

    // save RAM
    if(r_stored_iters > tot_mcmc_iters + 1){
      r_stored_iters = tot_mcmc_iters + 1;
    }

    int n_nu = alpha_3_samp.n_elem;

    double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    arma::vec pi_est = arma::zeros(pi_samp.n_rows);
    arma::mat Z_est = arma::zeros(Y.n_rows, Z_samp.n_cols);
    arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
    arma::cube eta_est = arma::zeros(eta_samp(0,0).n_rows, eta_samp(0,0).n_cols, eta_samp(0,0).n_slices);
    arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

    for(int i = 0; i < Z_est.n_cols; i++){
      pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
    }

    for(int i = 0; i < Z_est.n_cols; i++){
      for(int j = 0; j < Z_est.n_rows; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
        }
        Z_est(j,i) = arma::median(ph_Z);
      }
      for(int j = 0; j < nu_samp.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
        }
        nu_est(i,j) = arma::median(ph_nu);
        for(int d = 0; d < X1.n_cols; d++){
          for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
            ph_eta(l - std::round(n_nu * burnin_prop)) = eta_samp(l,0)(j,d,i);
          }
          eta_est(j,d,i) = arma::median(ph_eta);
        }
      }
    }

    // normalize
    for(int i = 0; i < Z_est.n_rows; i++){
      Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
    }

    pi_est = pi_est / arma::accu(pi_est);

    int n_Phi = sigma_samp.n_elem;

    double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
    arma::mat delta_est = arma::zeros(delta_samp.n_rows, delta_samp.n_cols);
    arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < delta_samp.n_cols; j++){
      for(int i = 0; i < delta_samp.n_rows; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,j,l);
        }
        delta_est(i, j) = arma::median(ph_delta);
      }
    }

    arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
    arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
    arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < Phi_est.n_rows; i++){
      for(int j = 0; j < Phi_est.n_cols; j++){
        for(int m = 0; m < Phi_est.n_slices; m++){
          for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
            ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);

            ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
          }
          Phi_est(i,j,m) = arma::median(ph_phi);
          gamma_est(i,j,m) = arma::median(ph_gamma);
        }
      }
    }

    arma::mat A_est = arma::zeros(A_samp.n_rows, A_samp.n_cols);
    arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int j = 0; j < A_samp.n_rows; j++){
      for(int i = 0; i < A_samp.n_cols; i++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(j, i, l);
        }
        A_est(j, i) = arma::median(ph_A);
      }
    }

    arma::vec tau_est = arma::zeros(tau_samp.n_cols);
    arma::mat tau_eta_est = arma::zeros(tau_eta_samp.n_rows, tau_eta_samp.n_cols);
    arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    arma::vec ph_tau_eta = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
    for(int i = 0; i < tau_est.n_elem; i++){
      for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
        ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
      }
      tau_est(i) = arma::median(ph_tau);
      for(int j = 0; j < tau_eta_est.n_cols; j++){
        for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
          ph_tau_eta(l - std::round(n_nu * burnin_prop)) = tau_eta_samp(i,j,l);
        }
        tau_eta_est(i,j) = arma::median(ph_tau_eta);
      }
    }
    arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
    arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
    for(int i = 0; i < chi_est.n_rows; i++){
      for(int j = 0; j < chi_est.n_cols; j++){
        for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
          ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
        }
        chi_est(i,j) = arma::median(ph_chi);
      }
    }
    if(covariance_adj == false){
      // start MCMC sampling
      Rcpp::List mod1 = BayesFMMM::BFMMM_MTT_warm_startMV_MeanAdj(Y, X1, thinning_num, K,
                                                                  n_eigen, tot_mcmc_iters,
                                                                  r_stored_iters, n_temp_trans,
                                                                  c1, b, nu_1, alpha1l, alpha2l,
                                                                  beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                                  var_alpha3, var_epsilon1,
                                                                  var_epsilon2, alpha_nu, beta_nu,
                                                                  alpha_eta, beta_eta, alpha_0,
                                                                  beta_0, dir1, beta_N_t, N_t,
                                                                  Z_est, pi_est, alpha_3_est,
                                                                  delta_est, gamma_est, Phi_est, A_est,
                                                                  nu_est, eta_est, tau_est, tau_eta_est,
                                                                  sigma_est, chi_est);
      mod2 =  Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                 Rcpp::Named("chi", mod1["chi"]),
                                 Rcpp::Named("pi", mod1["pi"]),
                                 Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                 Rcpp::Named("A", mod1["A"]),
                                 Rcpp::Named("delta", mod1["delta"]),
                                 Rcpp::Named("sigma_sq", mod1["sigma_sq"]),
                                 Rcpp::Named("tau", mod1["tau"]),
                                 Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                 Rcpp::Named("eta", mod1["eta"]),
                                 Rcpp::Named("gamma", mod1["gamma"]),
                                 Rcpp::Named("Phi", mod1["Phi"]),
                                 Rcpp::Named("Z", mod1["Z"]),
                                 Rcpp::Named("loglik", mod1["loglik"]));
    }else{
      arma::field<arma::cube> delta_xi_samp = theta_est["delta_xi"];
      arma::field<arma::cube> A_xi_samp = theta_est["A_xi"];
      arma::field<arma::cube> xi_samp1 = theta_est["xi"];
      arma::field<arma::cube> xi_samp(n_nu, K);
      for(int k = 0; k < K; k++){
        for(int n = 0; n < n_nu; n++){
          xi_samp(n,k) = xi_samp1(((k * n_nu) + n), 0);
        }
      }
      arma::field<arma::cube> gamma_xi_samp1 = theta_est["gamma_xi"];
      arma::field<arma::cube> gamma_xi_samp(n_nu, K);
      for(int k = 0; k < K; k++){
        for(int n = 0; n < n_nu; n++){
          gamma_xi_samp(n,k) = gamma_xi_samp1(((k * n_nu) + n), 0);
        }
      }

      arma::cube delta_xi_est = arma::zeros(delta_xi_samp(0,0).n_rows,
                                            delta_xi_samp(0,0).n_cols,
                                            delta_xi_samp(0,0).n_slices);
      arma::cube A_xi_est = arma::zeros(A_xi_samp(0,0).n_rows,
                                        A_xi_samp(0,0).n_cols,
                                        A_xi_samp(0,0).n_slices);

      arma::field<arma::cube> xi_est(1,K);
      arma::field<arma::cube> gamma_xi_est(1,K);
      for(int j = 0; j < K; j++){
        xi_est(0,j) = arma::zeros(xi_samp(0,j).n_rows, xi_samp(0,j).n_cols,
               xi_samp(0,j).n_slices);
        gamma_xi_est(0,j) = arma::zeros(gamma_xi_samp(0,j).n_rows,
                     gamma_xi_samp(0,j).n_cols, gamma_xi_samp(0,j).n_slices);
      }

      arma::vec delta_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec A_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
      arma::vec gamma_xi_ph = arma::zeros(n_nu - std::round(n_nu * burnin_prop));

      for(int j = 0; j < K; j++){
        for(int d = 0; d < X1.n_cols; d++){
          for(int i = 0; i < 2; i++){
            for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
              A_xi_ph(l - std::round(n_nu * burnin_prop)) = A_xi_samp(l,0)(j, i, d);
            }
            A_xi_est(j, i, d) = arma::median(A_xi_ph);
          }
          for(int m = 0; m < n_eigen; m++){
            for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
              delta_xi_ph(l - std::round(n_nu * burnin_prop)) = delta_xi_samp(l,0)(j, m, d);
            }
            delta_xi_est(j, m, d) = arma::median(delta_xi_ph);
            for(int p = 0; p < xi_samp(0,j).n_rows; p++){
              for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
                xi_ph(l - std::round(n_nu * burnin_prop)) = xi_samp(l,j)(p, d, m);
                gamma_xi_ph(l - std::round(n_nu * burnin_prop)) = gamma_xi_samp(l,j)(p, d, m);
              }
              xi_est(0,j)(p, d, m) = arma::median(xi_ph);
              gamma_xi_est(0,j)(p, d, m) = arma::median(gamma_xi_ph);
            }
          }
        }
      }

      Rcpp::List mod1 = BayesFMMM::BFMMM_MTT_warm_startMV_Mean_CovAdj(Y, X1, thinning_num, K,
                                                                      n_eigen, tot_mcmc_iters,
                                                                      r_stored_iters, n_temp_trans,
                                                                      c1, b, nu_1, alpha1l, alpha2l,
                                                                      beta1l, beta2l, a_Z_PM, a_pi_PM,
                                                                      var_alpha3, var_epsilon1,
                                                                      var_epsilon2, alpha_nu, beta_nu,
                                                                      alpha_eta, beta_eta, alpha_0,
                                                                      beta_0, dir1, beta_N_t, N_t,
                                                                      Z_est, pi_est, alpha_3_est,
                                                                      delta_est, delta_xi_est, gamma_est,
                                                                      gamma_xi_est, Phi_est, xi_est, A_est,
                                                                      A_xi_est, nu_est, eta_est, tau_est,
                                                                      tau_eta_est, sigma_est, chi_est);

      mod2 =  Rcpp::List::create(Rcpp::Named("nu", mod1["nu"]),
                                 Rcpp::Named("chi", mod1["chi"]),
                                 Rcpp::Named("pi", mod1["pi"]),
                                 Rcpp::Named("alpha_3", mod1["alpha_3"]),
                                 Rcpp::Named("A", mod1["A"]),
                                 Rcpp::Named("delta", mod1["delta"]),
                                 Rcpp::Named("sigma_sq", mod1["sigma_sq"]),
                                 Rcpp::Named("tau", mod1["tau"]),
                                 Rcpp::Named("tau_eta", mod1["tau_eta"]),
                                 Rcpp::Named("xi", mod1["xi"]),
                                 Rcpp::Named("delta_xi", mod1["delta_xi"]),
                                 Rcpp::Named("gamma_xi", mod1["gamma_xi"]),
                                 Rcpp::Named("A_xi", mod1["A_xi"]),
                                 Rcpp::Named("eta", mod1["eta"]),
                                 Rcpp::Named("gamma", mod1["gamma"]),
                                 Rcpp::Named("Phi", mod1["Phi"]),
                                 Rcpp::Named("Z", mod1["Z"]),
                                 Rcpp::Named("loglik", mod1["loglik"]));
    }
  }


  return mod2;
}
