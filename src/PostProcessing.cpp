#include <RcppArmadillo.h>
#include <splines2Armadillo.h>
#include <cmath>
#include <BayesFMMM.h>

//' Calculates the credible interval for the mean (Functional Data)
//'
//' This function calculates a credible interval with the user specified coverage.
//' In order to run this function, the directory of the posterior samples needs
//' to be specified. The function will return the credible intervals and the median
//' posterior estimate of the mean function at the time points specified by the
//' user (\code{time} variable). The user can specify if they would like the algorithm
//' to automatically rescale the parameters for interpretability (suggested). If
//' the user chooses to rescale, then all class memberships will be rescaled so
//' that at least one observation is in only one class. The user can also specify
//' if they want pointwise credible intervals or simultaneous credible intervals.
//' The simultaneous intervals will likely be wider than the pointwise credible
//' intervals.
//'
//' @name FMeanCI
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param time Vector containing time points of interest
//' @param basis_degree Int containing the degree of B-splines used
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param k Int containing the cluster group of which you want to get the credible interval for
//' @param alpha Double specifying the percentile of the credible interval ((1 - alpha) * 100 percent)
//' @param rescale Boolean indicating whether or not we should rescale the Z variables so that there is at least one observation almost completely in one group
//' @param simultaneous Boolean indicating whether or not the credible intervals should be simultaneous credible intervals or pointwise credible intervals
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix containing covariates at points of interest (of dimension W x D (number of points of interest x number of covariates))
//' @return CI list containing the credible interval for the mean function, as well as the median posterior estimate of the mean function. Posterior samples fo the mean function are also returned.
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{k}}{must be an integer larger than 1 and less than or equal to the number of clusters in the model}
//'   \item{\code{alpha}}{must be between 0 and 1}
//'   \item{\code{burnin_prop}}{must be less than 1 and greater than or equal to 0}
//'   \item{\code{X}}{must have the same number of columns as covariates in the model (D)}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' time <- seq(0, 990, 10)
//' k <- 2
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' ## Get CI for mean function
//' CI <- FMeanCI(dir, n_files, time, basis_degree, boundary_knots, internal_knots, k)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' time <- seq(0, 990, 10)
//' k <- 2
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' CI <- FMeanCI(dir, n_files, time, basis_degree, boundary_knots, internal_knots, k, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' time <- seq(0, 990, 10)
//' k <- 2
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' CI <- FMeanCI(dir, n_files, time, basis_degree, boundary_knots, internal_knots, k, X = X)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List FMeanCI(const std::string dir,
                   const int n_files,
                   const arma::vec time,
                   const int basis_degree,
                   const arma::vec boundary_knots,
                   const arma::vec internal_knots,
                   const int k,
                   const double alpha = 0.05,
                   bool rescale = true,
                   const bool simultaneous = false,
                   const double burnin_prop = 0.1,
                   Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue){
  // initialize object to return
  Rcpp::List CI;

  if(X.isNull()){
    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }

    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    if(k <= 0){
      Rcpp::stop("'k' must be positive");
    }
    if(k > nu_i.n_rows){
      Rcpp::stop("'k' must be less than or equal to the number of clusters in the model");
    }
    arma::cube nu_samp1 = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp1.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp1.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                       (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols,
                                     std::ceil((nu_i.n_slices * n_files)* (1 - burnin_prop)));
    nu_samp = nu_samp1.subcube(0, 0, std::floor(nu_i.n_slices * n_files * burnin_prop),
                               nu_samp1.n_rows-1, nu_samp1.n_cols-1, nu_samp1.n_slices-1);

    if(rescale == true){
      if(nu_i.n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }

    splines2::BSpline bspline;

    bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat{bspline.basis(true)};

    // Make B_obs
    arma::mat B = bspline_mat;
    arma::mat f_samp = arma::zeros(nu_samp.n_slices, time.n_elem);

    // Initialize placeholders
    arma::vec CI_Upper = arma::zeros(time.n_elem);
    arma::vec CI_50 = arma::zeros(time.n_elem);
    arma::vec CI_Lower = arma::zeros(time.n_elem);

    if(simultaneous == false){
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }
        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);

        // rescale Z and nu
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int l = 0; l < Z_samp.n_rows; l++){
              ph(l) = Z_samp(l,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          nu_samp.slice(j) = transform_mat * nu_samp.slice(j);
        }

        for(int i = 0; i < nu_samp.n_slices; i++){
          f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
        }
      } else{
        for(int i = 0; i < nu_samp.n_slices; i++){
          f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
        }
      }

      arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
      arma::vec q = arma::zeros(3);

      for(int i = 0; i < time.n_elem; i++){
        q = arma::quantile(f_samp.col(i), p);
        CI_Lower(i) = q(0);
        CI_50(i) = q(1);
        CI_Upper(i) = q(2);
      }
    }else{
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and nu
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int l = 0; l < Z_samp.n_rows; l++){
              ph(l) = Z_samp(l,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          nu_samp.slice(j) = transform_mat * nu_samp.slice(j);
        }

        for(int i = 0; i < nu_samp.n_slices; i++){
          f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
        }
        arma::rowvec f_mean = arma::mean(f_samp, 0);
        arma::rowvec f_sd = arma::stddev(f_samp, 0, 0);

        arma::vec C = arma::zeros(nu_samp.n_slices);
        arma::vec ph1 = arma::zeros(time.n_elem);
        for(int i = 0; i < nu_samp.n_slices; i++){
          for(int j = 0; j < time.n_elem; j++){
            ph1(j) = std::abs((f_samp(i,j) - f_mean(j)) / f_sd(j));
          }
          C(i) = arma::max(ph1);
        }

        arma::vec p = {1 - alpha};
        arma::vec q = arma::zeros(1);
        q = arma::quantile(C, p);

        for(int i = 0; i < time.n_elem; i++){
          CI_Lower(i) = f_mean(i) - q(0) * f_sd(i);
          CI_50(i) = f_mean(i);
          CI_Upper(i) =  f_mean(i) + q(0) * f_sd(i);
        }
      }else{
        for(int i = 0; i < nu_samp.n_slices; i++){
          f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
        }
        arma::rowvec f_mean = arma::mean(f_samp, 0);
        arma::rowvec f_sd = arma::stddev(f_samp, 0, 0);

        arma::vec C = arma::zeros(nu_samp.n_slices);
        arma::vec ph1 = arma::zeros(time.n_elem);
        for(int i = 0; i < nu_samp.n_slices; i++){
          for(int j = 0; j < time.n_elem; j++){
            ph1(j) = std::abs((f_samp(i,j) - f_mean(j)) / f_sd(j));
          }
          C(i) = arma::max(ph1);
        }

        arma::vec p = {1 - alpha};
        arma::vec q = arma::zeros(1);
        q = arma::quantile(C, p);

        for(int i = 0; i < time.n_elem; i++){
          CI_Lower(i) = f_mean(i) - q(0) * f_sd(i);
          CI_50(i) = f_mean(i);
          CI_Upper(i) =  f_mean(i) + q(0) * f_sd(i);
        }
      }
    }

    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("mean_trace", f_samp));
    return(CI);
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);
    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }

    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    if(k <= 0){
      Rcpp::stop("'k' must be positive");
    }
    if(k > nu_i.n_rows){
      Rcpp::stop("'k' must be less than or equal to the number of clusters in the model");
    }
    arma::cube nu_samp1 = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp1.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp1.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                       (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols,
                                     std::ceil((nu_i.n_slices * n_files)* (1 - burnin_prop)));
    nu_samp = nu_samp1.subcube(0, 0, std::floor(nu_i.n_slices * n_files * burnin_prop),
                               nu_samp1.n_rows-1, nu_samp1.n_cols-1, nu_samp1.n_slices-1);

    if(rescale == true){
      if(nu_i.n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }

    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    if(X1.n_cols != eta_i(0,0).n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    arma::field<arma::cube> eta_samp1(nu_samp1.n_slices, 1);
    for(int l = 0; l < nu_i.n_slices; l++){
      eta_samp1(l,0) = eta_i(l,0);
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int l = 0; l < nu_i.n_slices; l++){
        eta_samp1((i * nu_i.n_slices) + l, 0) = eta_i(l,0);
      }
    }

    arma::field<arma::cube> eta_samp(nu_samp.n_slices, 1);
    for(int i = 0; i < std::round(nu_samp.n_slices); i++){
      eta_samp(i,0) = eta_samp1(i + std::floor(nu_samp1.n_slices * burnin_prop), 0);
    }

    splines2::BSpline bspline;

    bspline = splines2::BSpline(time, internal_knots, basis_degree,
                                boundary_knots);
    // Get Basis matrix (100 x 8)
    arma::mat bspline_mat{bspline.basis(true)};

    // Make B_obs
    arma::mat B = bspline_mat;
    arma::cube f_samp = arma::zeros(X1.n_rows, time.n_elem, nu_samp.n_slices);

    // Initialize placeholders
    arma::mat CI_Upper = arma::zeros(X1.n_rows, time.n_elem);
    arma::mat CI_50 = arma::zeros(X1.n_rows, time.n_elem);
    arma::mat CI_Lower = arma::zeros(X1.n_rows, time.n_elem);

    if(simultaneous == false){
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }
        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);

        // rescale Z and nu
        arma::mat eta_ph = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int l = 0; l < Z_samp.n_rows; l++){
              ph(l) = Z_samp(l,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          nu_samp.slice(j) = transform_mat * nu_samp.slice(j);

          // transform eta parameters
          for(int d = 0; d < X1.n_cols; d++){
            for(int r = 0; r < nu_samp.n_rows; r++){
              for(int p = 0; p < nu_samp.n_cols; p++){
                eta_ph(r,p) = eta_samp(j,0)(p,d,r);
              }
            }
            eta_ph = transform_mat * eta_ph;
            for(int r = 0; r < nu_samp.n_rows; r++){
              for(int p = 0; p < nu_samp.n_cols; p++){
                eta_samp(j,0)(p,d,r) = eta_ph(r,p);
              }
            }
          }
        }
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            f_samp.slice(i).row(n) = (B * (nu_samp.slice(i).row(k-1).t() +
              (eta_samp(i,0).slice(k-1) * X1.row(n).t()))).t();
          }
        }
      } else{
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            f_samp.slice(i).row(n) = (B * (nu_samp.slice(i).row(k-1).t() +
              (eta_samp(i,0).slice(k-1) * X1.row(n).t()))).t();
          }
        }
      }

      arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
      arma::vec q = arma::zeros(3);
      arma::vec f_ph = arma::zeros(nu_samp.n_slices);
      for(int n = 0; n < X1.n_rows; n++){
        for(int i = 0; i < time.n_elem; i++){
          for(int r = 0; r < nu_samp.n_slices; r++){
            f_ph(r) = f_samp(n,i,r);
          }
          q = arma::quantile(f_ph, p);
          CI_Lower(n,i) = q(0);
          CI_50(n,i) = q(1);
          CI_Upper(n,i) = q(2);
        }
      }
    }else{
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and nu
        arma::mat eta_ph = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int l = 0; l < Z_samp.n_rows; l++){
              ph(l) = Z_samp(l,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          nu_samp.slice(j) = transform_mat * nu_samp.slice(j);

          // transform eta parameters
          for(int d = 0; d < X1.n_cols; d++){
            for(int r = 0; r < nu_samp.n_rows; r++){
              for(int p = 0; p < nu_samp.n_cols; p++){
                eta_ph(r,p) = eta_samp(j,0)(p,d,r);
              }
            }
            eta_ph = transform_mat * eta_ph;
            for(int r = 0; r < nu_samp.n_rows; r++){
              for(int p = 0; p < nu_samp.n_cols; p++){
                eta_samp(j,0)(p,d,r) = eta_ph(r,p);
              }
            }
          }
        }

        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            f_samp.slice(i).row(n) = (B * (nu_samp.slice(i).row(k-1).t() +
              (eta_samp(i,0).slice(k-1) * X1.row(n).t()))).t();
          }
        }
        arma::mat f_mean_ph = arma::zeros(nu_samp.n_slices, time.n_elem);
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            for(int j = 0; j < f_samp.n_cols; j++){
              f_mean_ph(i,j) = f_samp(n, j, i);
            }
          }
          arma::rowvec f_mean = arma::mean(f_mean_ph, 0);
          arma::rowvec f_sd = arma::stddev(f_mean_ph, 0, 0);

          arma::vec C = arma::zeros(nu_samp.n_slices);
          arma::vec ph1 = arma::zeros(time.n_elem);
          for(int i = 0; i < nu_samp.n_slices; i++){
            for(int j = 0; j < time.n_elem; j++){
              ph1(j) = std::abs((f_samp(n,j,i) - f_mean(j)) / f_sd(j));
            }
            C(i) = arma::max(ph1);
          }

          arma::vec p = {1 - alpha};
          arma::vec q = arma::zeros(1);
          q = arma::quantile(C, p);

          for(int i = 0; i < time.n_elem; i++){
            CI_Lower(n,i) = f_mean(i) - q(0) * f_sd(i);
            CI_50(n,i) = f_mean(i);
            CI_Upper(n,i) =  f_mean(i) + q(0) * f_sd(i);
          }
        }

      }else{
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            f_samp.slice(i).row(n) = (B * (nu_samp.slice(i).row(k-1).t() +
              (eta_samp(i,0).slice(k-1) * X1.row(n).t()))).t();
          }
        }
        arma::mat f_mean_ph = arma::zeros(nu_samp.n_slices, time.n_elem);
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            for(int j = 0; j < f_samp.n_cols; j++){
              f_mean_ph(i,j) = f_samp(n, j, i);
            }
          }
          arma::rowvec f_mean = arma::mean(f_mean_ph, 0);
          arma::rowvec f_sd = arma::stddev(f_mean_ph, 0, 0);

          arma::vec C = arma::zeros(nu_samp.n_slices);
          arma::vec ph1 = arma::zeros(time.n_elem);
          for(int i = 0; i < nu_samp.n_slices; i++){
            for(int j = 0; j < time.n_elem; j++){
              ph1(j) = std::abs((f_samp(n,j,i) - f_mean(j)) / f_sd(j));
            }
            C(i) = arma::max(ph1);
          }

          arma::vec p = {1 - alpha};
          arma::vec q = arma::zeros(1);
          q = arma::quantile(C, p);

          for(int i = 0; i < time.n_elem; i++){
            CI_Lower(n,i) = f_mean(i) - q(0) * f_sd(i);
            CI_50(n,i) = f_mean(i);
            CI_Upper(n,i) =  f_mean(i) + q(0) * f_sd(i);
          }
        }
      }
    }

    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("mean_trace", f_samp));
    return(CI);
  }

}


//' Calculates the credible interval for the mean (High Dimensional Functional Data)
//'
//' This function calculates a credible interval with the user specified coverage.
//' In order to run this function, the directory of the posterior samples needs
//' to be specified. The function will return the credible intervals and the median
//' posterior estimate of the mean function at the time points specified by the
//' user (\code{time} variable). The user can specify if they would like the algorithm
//' to automatically rescale the parameters for interpretability (suggested). If
//' the user chooses to rescale, then all class memberships will be rescaled so
//' that at least one observation is in only one class. The user can also specify
//' if they want pointwise credible intervals or simultaneous credible intervals.
//' The simultaneous intervals will likely be wider than the pointwise credible
//' intervals.
//'
//' @name HDFMeanCI
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param time List of matrices that contain the observed time points (each column is a dimension)
//' @param basis_degree Vector containing the desired basis degree for each dimension
//' @param boundary_knots Matrix containing the boundary knots for each dimension (each row is a dimension)
//' @param internal_knots List of vectors containing the internal knots for each dimension
//' @param k Int containing the cluster group of which you want to get the credible interval for
//' @param alpha Double specifying the percentile of the credible interval ((1 - alpha) * 100 percent)
//' @param rescale Boolean indicating whether or not we should rescale the Z variables so that there is at least one observation almost completely in one group
//' @param simultaneous Boolean indicating whether or not the credible intervals should be simultaneous credible intervals or pointwise credible intervals
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix containing covariates at points of interest (of dimension W x D (number of points of interest x number of covariates))
//' @return CI list containing the credible interval for the mean function, as well as the median posterior estimate of the mean function. Also returns posterior samples of the mean function.
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{basis_degree}}{each element must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of corresponding \code{boundary_knots}}
//'   \item{\code{k}}{must be an integer larger than 1 and less than or equal to the number of clusters in the model}
//'   \item{\code{alpha}}{must be between 0 and 1}
//'   \item{\code{burnin_prop}}{must be less than 1 and greater than or equal to 0}
//'   \item{\code{X}}{must have the same number of columns as covariates in the model (D)}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "HDFunctional_trace", "", package = "BayesFMMM")
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//' time <- time[[1]]
//' n_files <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' ## Get CI for mean function
//' CI <- HDFMeanCI(dir, n_files, time, basis_degree, boundary_knots, internal_knots, k)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' dir <- system.file("test-data", "HDFunctional_trace", "", package = "BayesFMMM")
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//' time <- time[[1]]
//' n_files <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' CI <- HDFMeanCI(dir, n_files, time, basis_degree, boundary_knots, internal_knots, k, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' dir <- system.file("test-data", "HDFunctional_trace", "", package = "BayesFMMM")
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//' time <- time[[1]]
//' n_files <- 1
//' K <- 2
//' n_funct <- 20
//' basis_degree <- c(2,2)
//' n_eigen <- 2
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' CI <- HDFMeanCI(dir, n_files, time, basis_degree, boundary_knots, internal_knots, k, X = X)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List HDFMeanCI(const std::string dir,
                     const int n_files,
                     const arma::mat time,
                     const arma::vec basis_degree,
                     const arma::mat boundary_knots,
                     const arma::field<arma::vec> internal_knots,
                     const int k,
                     const double alpha = 0.05,
                     bool rescale = true,
                     const bool simultaneous = false,
                     const double burnin_prop = 0.1,
                     Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue){
  Rcpp::List CI;

  if(X.isNull()){
    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
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

    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    if(k <= 0){
      Rcpp::stop("'k' must be positive");
    }
    if(k > nu_i.n_rows){
      Rcpp::stop("'k' must be less than or equal to the number of clusters in the model");
    }
    arma::cube nu_samp1 = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp1.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp1.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                       (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols,
                                     std::ceil((nu_i.n_slices * n_files)* (1 - burnin_prop)));
    nu_samp = nu_samp1.subcube(0, 0, std::floor(nu_i.n_slices * n_files * burnin_prop),
                               nu_samp1.n_rows-1, nu_samp1.n_cols-1, nu_samp1.n_slices-1);
    arma::field<arma::mat> time1(1,1);
    time1(0,0) = time;
    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time1, 1, basis_degree,
                                                            boundary_knots, internal_knots);
    arma::mat B = B_obs(0,0);

    arma::mat f_samp = arma::zeros(nu_samp.n_slices, time.n_rows);

    if(rescale == true){
      if(nu_i.n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }


    // Initialize placeholders
    arma::vec CI_Upper = arma::zeros(time.n_rows);
    arma::vec CI_50 = arma::zeros(time.n_rows);
    arma::vec CI_Lower = arma::zeros(time.n_rows);

    if(simultaneous == false){
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }
        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);

        // rescale Z and nu
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int l = 0; l < Z_samp.n_rows; l++){
              ph(l) = Z_samp(l,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          nu_samp.slice(j) = transform_mat * nu_samp.slice(j);
        }

        for(int i = 0; i < nu_samp.n_slices; i++){
          f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
        }
      } else{
        for(int i = 0; i < nu_samp.n_slices; i++){
          f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
        }
      }

      arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
      arma::vec q = arma::zeros(3);

      for(int i = 0; i < time.n_rows; i++){
        q = arma::quantile(f_samp.col(i), p);
        CI_Lower(i) = q(0);
        CI_50(i) = q(1);
        CI_Upper(i) = q(2);
      }
    }else{
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and nu
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int l = 0; l < Z_samp.n_rows; l++){
              ph(l) = Z_samp(l,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          nu_samp.slice(j) = transform_mat * nu_samp.slice(j);
        }
        arma::mat f_samp = arma::zeros(nu_samp.n_slices, time.n_rows);
        for(int i = 0; i < nu_samp.n_slices; i++){
          f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
        }
        arma::rowvec f_mean = arma::mean(f_samp, 0);
        arma::rowvec f_sd = arma::stddev(f_samp, 0, 0);

        arma::vec C = arma::zeros(nu_samp.n_slices);
        arma::vec ph1 = arma::zeros(time.n_rows);
        for(int i = 0; i < nu_samp.n_slices; i++){
          for(int j = 0; j < time.n_rows; j++){
            ph1(j) = std::abs((f_samp(i,j) - f_mean(j)) / f_sd(j));
          }
          C(i) = arma::max(ph1);
        }

        arma::vec p = {1 - alpha};
        arma::vec q = arma::zeros(1);
        q = arma::quantile(C, p);

        for(int i = 0; i < time.n_rows; i++){
          CI_Lower(i) = f_mean(i) - q(0) * f_sd(i);
          CI_50(i) = f_mean(i);
          CI_Upper(i) =  f_mean(i) + q(0) * f_sd(i);
        }
      }else{
        arma::mat f_samp = arma::zeros(nu_samp.n_slices, time.n_rows);
        for(int i = 0; i < nu_samp.n_slices; i++){
          f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
        }
        arma::rowvec f_mean = arma::mean(f_samp, 0);
        arma::rowvec f_sd = arma::stddev(f_samp, 0, 0);

        arma::vec C = arma::zeros(nu_samp.n_slices);
        arma::vec ph1 = arma::zeros(time.n_rows);
        for(int i = 0; i < nu_samp.n_slices; i++){
          for(int j = 0; j < time.n_rows; j++){
            ph1(j) = std::abs((f_samp(i,j) - f_mean(j)) / f_sd(j));
          }
          C(i) = arma::max(ph1);
        }

        arma::vec p = {1 - alpha};
        arma::vec q = arma::zeros(1);
        q = arma::quantile(C, p);

        for(int i = 0; i < time.n_rows; i++){
          CI_Lower(i) = f_mean(i) - q(0) * f_sd(i);
          CI_50(i) = f_mean(i);
          CI_Upper(i) =  f_mean(i) + q(0) * f_sd(i);
        }
      }
    }

    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("mean_trace", f_samp));
    return(CI);
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
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

    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    if(k <= 0){
      Rcpp::stop("'k' must be positive");
    }
    if(k > nu_i.n_rows){
      Rcpp::stop("'k' must be less than or equal to the number of clusters in the model");
    }
    arma::cube nu_samp1 = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp1.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp1.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                       (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols,
                                     std::ceil((nu_i.n_slices * n_files)* (1 - burnin_prop)));
    nu_samp = nu_samp1.subcube(0, 0, std::floor(nu_i.n_slices * n_files * burnin_prop),
                               nu_samp1.n_rows-1, nu_samp1.n_cols-1, nu_samp1.n_slices-1);

    arma::field<arma::mat> time1(1,1);
    time1(0,0) = time;
    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time1, 1, basis_degree,
                                                            boundary_knots, internal_knots);
    arma::mat B = B_obs(0,0);

    arma::cube f_samp = arma::zeros(X1.n_rows, time.n_rows, nu_samp.n_slices);

    if(rescale == true){
      if(nu_i.n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }

    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    if(X1.n_cols != eta_i(0,0).n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    arma::field<arma::cube> eta_samp1(nu_samp1.n_slices, 1);
    for(int l = 0; l < nu_i.n_slices; l++){
      eta_samp1(l,0) = eta_i(l,0);
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int l = 0; l < nu_i.n_slices; l++){
        eta_samp1((i * nu_i.n_slices) + l, 0) = eta_i(l,0);
      }
    }

    arma::field<arma::cube> eta_samp(nu_samp.n_slices, 1);
    for(int i = 0; i < std::round(nu_samp.n_slices); i++){
      eta_samp(i,0) = eta_samp1(i + std::floor(nu_samp1.n_slices * burnin_prop), 0);
    }

    // Initialize placeholders
    arma::mat CI_Upper = arma::zeros(X1.n_rows, time.n_rows);
    arma::mat CI_50 = arma::zeros(X1.n_rows, time.n_rows);
    arma::mat CI_Lower = arma::zeros(X1.n_rows, time.n_rows);

    if(simultaneous == false){
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }
        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);

        // rescale Z and nu
        arma::mat eta_ph = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int l = 0; l < Z_samp.n_rows; l++){
              ph(l) = Z_samp(l,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          nu_samp.slice(j) = transform_mat * nu_samp.slice(j);

          // transform eta parameters
          for(int d = 0; d < X1.n_cols; d++){
            for(int r = 0; r < nu_samp.n_rows; r++){
              for(int p = 0; p < nu_samp.n_cols; p++){
                eta_ph(r,p) = eta_samp(j,0)(p,d,r);
              }
            }
            eta_ph = transform_mat * eta_ph;
            for(int r = 0; r < nu_samp.n_rows; r++){
              for(int p = 0; p < nu_samp.n_cols; p++){
                eta_samp(j,0)(p,d,r) = eta_ph(r,p);
              }
            }
          }
        }

        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            f_samp.slice(i).row(n) = (B * (nu_samp.slice(i).row(k-1).t() +
              (eta_samp(i,0).slice(k-1) * X1.row(n).t()))).t();
          }
        }
      } else{
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            f_samp.slice(i).row(n) = (B * (nu_samp.slice(i).row(k-1).t() +
              (eta_samp(i,0).slice(k-1) * X1.row(n).t()))).t();
          }
        }
      }

      arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
      arma::vec q = arma::zeros(3);
      arma::vec f_ph = arma::zeros(nu_samp.n_slices);

      for(int n = 0; n < X1.n_rows; n++){
        for(int i = 0; i < time.n_rows; i++){
          for(int r = 0; r < nu_samp.n_slices; r++){
            f_ph(r) = f_samp(n,i,r);
          }
          q = arma::quantile(f_ph, p);
          CI_Lower(n,i) = q(0);
          CI_50(n,i) = q(1);
          CI_Upper(n,i) = q(2);
        }
      }
    }else{
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and nu
        arma::mat eta_ph = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int l = 0; l < Z_samp.n_rows; l++){
              ph(l) = Z_samp(l,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          nu_samp.slice(j) = transform_mat * nu_samp.slice(j);

          // transform eta parameters
          for(int d = 0; d < X1.n_cols; d++){
            for(int r = 0; r < nu_samp.n_rows; r++){
              for(int p = 0; p < nu_samp.n_cols; p++){
                eta_ph(r,p) = eta_samp(j,0)(p,d,r);
              }
            }
            eta_ph = transform_mat * eta_ph;
            for(int r = 0; r < nu_samp.n_rows; r++){
              for(int p = 0; p < nu_samp.n_cols; p++){
                eta_samp(j,0)(p,d,r) = eta_ph(r,p);
              }
            }
          }
        }

        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            f_samp.slice(i).row(n) = (B * (nu_samp.slice(i).row(k-1).t() +
              (eta_samp(i,0).slice(k-1) * X1.row(n).t()))).t();
          }
        }


        arma::mat f_mean_ph = arma::zeros(nu_samp.n_slices, time.n_rows);
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            for(int j = 0; j < f_samp.n_cols; j++){
              f_mean_ph(i,j) = f_samp(n, j, i);
            }
          }
          arma::rowvec f_mean = arma::mean(f_mean_ph, 0);
          arma::rowvec f_sd = arma::stddev(f_mean_ph, 0, 0);

          arma::vec C = arma::zeros(nu_samp.n_slices);
          arma::vec ph1 = arma::zeros(time.n_rows);
          for(int i = 0; i < nu_samp.n_slices; i++){
            for(int j = 0; j < time.n_rows; j++){
              ph1(j) = std::abs((f_samp(n,j,i) - f_mean(j)) / f_sd(j));
            }
            C(i) = arma::max(ph1);
          }

          arma::vec p = {1 - alpha};
          arma::vec q = arma::zeros(1);
          q = arma::quantile(C, p);

          for(int i = 0; i < time.n_rows; i++){
            CI_Lower(n,i) = f_mean(i) - q(0) * f_sd(i);
            CI_50(n,i) = f_mean(i);
            CI_Upper(n,i) =  f_mean(i) + q(0) * f_sd(i);
          }
        }

      }else{
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            f_samp.slice(i).row(n) = (B * (nu_samp.slice(i).row(k-1).t() +
              (eta_samp(i,0).slice(k-1) * X1.row(n).t()))).t();
          }
        }
        arma::mat f_mean_ph = arma::zeros(nu_samp.n_slices, time.n_rows);
        for(int n = 0; n < X1.n_rows; n++){
          for(int i = 0; i < nu_samp.n_slices; i++){
            for(int j = 0; j < f_samp.n_cols; j++){
              f_mean_ph(i,j) = f_samp(n, j, i);
            }
          }
          arma::rowvec f_mean = arma::mean(f_mean_ph, 0);
          arma::rowvec f_sd = arma::stddev(f_mean_ph, 0, 0);

          arma::vec C = arma::zeros(nu_samp.n_slices);
          arma::vec ph1 = arma::zeros(time.n_rows);
          for(int i = 0; i < nu_samp.n_slices; i++){
            for(int j = 0; j < time.n_rows; j++){
              ph1(j) = std::abs((f_samp(n,j,i) - f_mean(j)) / f_sd(j));
            }
            C(i) = arma::max(ph1);
          }

          arma::vec p = {1 - alpha};
          arma::vec q = arma::zeros(1);
          q = arma::quantile(C, p);

          for(int i = 0; i < time.n_rows; i++){
            CI_Lower(n,i) = f_mean(i) - q(0) * f_sd(i);
            CI_50(n,i) = f_mean(i);
            CI_Upper(n,i) =  f_mean(i) + q(0) * f_sd(i);
          }
        }
      }
    }

    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("mean_trace", f_samp));
    return(CI);
  }

}

//' Calculates the credible interval for the mean (Multivariate Data)
//'
//' This function calculates a credible interval with the user specified coverage.
//' In order to run this function, the directory of the posterior samples needs
//' to be specified. The function will return the credible intervals and the median
//' posterior estimate of the mean. The user can specify if they would like the
//' algorithm to automatically rescale the parameters for interpretability
//' (suggested). If the user chooses to rescale, then all class memberships will
//' be rescaled so that at least one observation is in only one class.
//'
//' @name MVMeanCI
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param alpha Double specifying the percentile of the credible interval ((1 - alpha) * 100 percent)
//' @param rescale Boolean indicating whether or not we should rescale the Z variables so that there is at least one observation almost completely in one group
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix containing covariates at points of interest (of dimension W x D (number of points of interest x number of covariates))
//' @return CI list containing the credible interval for the mean function, as well as the median posterior estimate of the mean function. Posterior draws of the mean structure are also returned. If covariate adjusted, the third index corresponds number of points of interest
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{alpha}}{must be between 0 and 1}
//'   \item{\code{burnin_prop}}{must be less than 1 and greater than or equal to 0}
//'   \item{\code{X}}{must have the same number of columns as covariates in the model (D)}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//'
//' ## Get CI for mean function
//' CI <- MVMeanCI(dir, n_files)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' CI <- MVMeanCI(dir, n_files, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' CI <- MVMeanCI(dir, n_files, X = X)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List MVMeanCI(const std::string dir,
                    const int n_files,
                    const double alpha = 0.05,
                    bool rescale = true,
                    const double burnin_prop = 0.1,
                    Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue){
  Rcpp::List CI;

  if(X.isNull()){
    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    arma::cube nu_samp1 = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp1.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp1.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                       (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols,
                                     std::ceil((nu_i.n_slices * n_files)* (1 - burnin_prop)));
    nu_samp = nu_samp1.subcube(0, 0, std::floor(nu_i.n_slices * n_files * burnin_prop),
                               nu_samp1.n_rows-1, nu_samp1.n_cols-1, nu_samp1.n_slices-1);

    if(rescale == true){
      if(nu_i.n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }

    // Initialize placeholders
    arma::mat CI_Upper = arma::zeros(nu_i.n_rows, nu_i.n_cols);
    arma::mat CI_50 = arma::zeros(nu_i.n_rows, nu_i.n_cols);
    arma::mat CI_Lower = arma::zeros(nu_i.n_rows, nu_i.n_cols);

    if(rescale == true){
      // Get Z matrix
      arma::cube Z_i;
      Z_i.load(dir + "Z0.txt");
      arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
      Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
      for(int i = 1; i < n_files; i++){
        Z_i.load(dir + "Z" + std::to_string(i) +".txt");
        Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
      }
      arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                      std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
      Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                               Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);

      // rescale Z and nu
      arma::mat transform_mat;
      arma::vec ph = arma::zeros(Z_samp.n_rows);
      for(int j = 0; j < Z_samp.n_slices; j++){
        transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
        int max_ind = 0;
        for(int i = 0; i < Z_samp.n_cols; i++){
          for(int l = 0; l < Z_samp.n_rows; l++){
            ph(l) = Z_samp(l,i,j);
          }
          max_ind = arma::index_max(ph);
          transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
        }
        nu_samp.slice(j) = transform_mat * nu_samp.slice(j);
      }
    }

    arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
    arma::vec q = arma::zeros(3);
    arma::vec ph = arma::zeros(nu_samp.n_slices);

    for(int i = 0; i < nu_i.n_cols; i++){
      for(int k = 0; k < nu_i.n_rows; k++){
        for(int n = 0; n < nu_samp.n_slices; n++){
          ph(n) = nu_samp(k,i,n);
        }
        q = arma::quantile(ph, p);
        CI_Lower(k, i) = q(0);
        CI_50(k,i) = q(1);
        CI_Upper(k,i) = q(2);
      }
    }

    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("mean_trace", nu_samp));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    arma::cube nu_samp1 = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp1.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp1.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                       (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols,
                                     std::ceil((nu_i.n_slices * n_files)* (1 - burnin_prop)));
    nu_samp = nu_samp1.subcube(0, 0, std::floor(nu_i.n_slices * n_files * burnin_prop),
                               nu_samp1.n_rows-1, nu_samp1.n_cols-1, nu_samp1.n_slices-1);

    if(rescale == true){
      if(nu_i.n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }

    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    if(X1.n_cols != eta_i(0,0).n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    arma::field<arma::cube> eta_samp1(nu_samp1.n_slices, 1);
    for(int l = 0; l < nu_i.n_slices; l++){
      eta_samp1(l,0) = eta_i(l,0);
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int l = 0; l < nu_i.n_slices; l++){
        eta_samp1((i * nu_i.n_slices) + l, 0) = eta_i(l,0);
      }
    }

    arma::field<arma::cube> eta_samp(nu_samp.n_slices, 1);
    for(int i = 0; i < std::round(nu_samp.n_slices); i++){
      eta_samp(i,0) = eta_samp1(i + std::floor(nu_samp1.n_slices * burnin_prop), 0);
    }

    // Initialize placeholders
    arma::cube CI_Upper = arma::zeros(nu_i.n_rows, nu_i.n_cols, X1.n_rows);
    arma::cube CI_50 = arma::zeros(nu_i.n_rows, nu_i.n_cols, X1.n_rows);
    arma::cube CI_Lower = arma::zeros(nu_i.n_rows, nu_i.n_cols, X1.n_rows);
    arma::field<arma::cube> mean_samp(X1.n_rows, 1);
    arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
    arma::vec q = arma::zeros(3);
    arma::vec ph = arma::zeros(nu_samp.n_slices);

    if(rescale == true){
      // Get Z matrix
      arma::cube Z_i;
      Z_i.load(dir + "Z0.txt");
      arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
      Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
      for(int i = 1; i < n_files; i++){
        Z_i.load(dir + "Z" + std::to_string(i) +".txt");
        Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
      }
      arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                      std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
      Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                               Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);

      // rescale Z and nu
      arma::mat eta_ph = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
      arma::mat transform_mat;
      arma::vec ph = arma::zeros(Z_samp.n_rows);
      for(int j = 0; j < Z_samp.n_slices; j++){
        transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
        int max_ind = 0;
        for(int i = 0; i < Z_samp.n_cols; i++){
          for(int l = 0; l < Z_samp.n_rows; l++){
            ph(l) = Z_samp(l,i,j);
          }
          max_ind = arma::index_max(ph);
          transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
        }
        nu_samp.slice(j) = transform_mat * nu_samp.slice(j);
        // transform eta parameters
        for(int d = 0; d < X1.n_cols; d++){
          for(int r = 0; r < nu_samp.n_rows; r++){
            for(int p = 0; p < nu_samp.n_cols; p++){
              eta_ph(r,p) = eta_samp(j,0)(p,d,r);
            }
          }
          eta_ph = transform_mat * eta_ph;
          for(int r = 0; r < nu_samp.n_rows; r++){
            for(int p = 0; p < nu_samp.n_cols; p++){
              eta_samp(j,0)(p,d,r) = eta_ph(r,p);
            }
          }
        }
      }
    }
    for(int j = 0; j < X1.n_rows; j++){
      mean_samp(j,0) = arma::zeros(nu_samp.n_rows, nu_samp.n_cols, nu_samp.n_slices);
      for(int i = 0; i < nu_samp.n_slices; i++){
        for(int k= 0; k < eta_samp(j,0).n_slices; k++){
          mean_samp(j,0).slice(i).row(k) = nu_samp.slice(i).row(k) + (eta_samp(i,0).slice(k) * X1.row(j).t()).t();
        }
      }

      for(int i = 0; i < nu_i.n_cols; i++){
        for(int k = 0; k < nu_i.n_rows; k++){
          for(int n = 0; n < nu_samp.n_slices; n++){
            ph(n) = mean_samp(j,0)(k,i,n);
          }
          q = arma::quantile(ph, p);
          CI_Lower(k, i, j) = q(0);
          CI_50(k, i, j) = q(1);
          CI_Upper(k, i, j) = q(2);
        }
      }
    }

    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("mean_trace", mean_samp));

  }


  return(CI);
}

//' Calculates the credible interval for the covariance (Functional Data)
//'
//' This function calculates a credible interval for the covariance function
//' between the l-th and m-th clusters, with the user specified coverage. This
//' function can handle covariate adjusted models, where the mean and covariance
//' functions depend on the covariates of interest. If not covariate adjusted, or
//' if the covariates only influence the mean structure, DO NOT specify \code{X} in
//' this function.
//' In order to run this function, the directory of the posterior samples needs
//' to be specified. The function will return the credible intervals and the median
//' posterior estimate of the covariance function at the time points specified by the
//' user (\code{time} variable). The user can specify if they would like the algorithm
//' to automatically rescale the parameters for interpretability (suggested). If
//' the user chooses to rescale, then all class memberships will be rescaled so
//' that at least one observation is in only one class. The user can also specify
//' if they want pointwise credible intervals or simultaneous credible intervals.
//' The simultaneous intervals will likely be wider than the pointwise credible
//' intervals.
//'
//' @name FCovCI
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param time1 Vector containing time points of interest for first cluster
//' @param time2 Vector containing time points of interest for second cluster
//' @param basis_degree Int containing the degree of B-splines used
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param l Int containing the 1st cluster group of which you want to get the credible interval for
//' @param m Int containing the 2nd cluster group of which you want to get the credible interval for
//' @param alpha Double specifying the percentile of the credible interval ((1 - alpha) * 100 percent)
//' @param rescale Boolean indicating whether or not we should rescale the Z variables so that there is at least one observation almost completely in one group
//' @param simultaneous Boolean indicating whether or not the credible intervals should be simultaneous credible intervals or pointwise credible intervals
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix containing covariates at points of interest (of dimension W x D (number of points of interest x number of covariates))
//' @return CI list containing the credible interval for the covariance function, as well as the median posterior estimate of the covariance function. Posterior estimates of the covariance function are also returned.
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{n_MCMC}}{must be an integer larger than or equal to 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{l}}{must be an integer larger than 1 and less than or equal to the number of clusters in the model}
//'   \item{\code{m}}{must be an integer larger than 1 and less than or equal to the number of clusters in the model}
//'   \item{\code{alpha}}{must be between 0 and 1}
//'   \item{\code{burnin_prop}}{must be less than 1 and greater than or equal to 0}
//'   \item{\code{X}}{must have the same number of columns as covariates in the model (D)}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' time1 <- seq(0, 990, 10)
//' time2 <- seq(0, 990, 10)
//' l <- 1
//' m <- 1
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' ## Get CI for Covaraince function
//' CI <- FCovCI(dir, n_files, time1, time2, basis_degree,
//'              boundary_knots, internal_knots, l, m)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' time1 <- seq(0, 990, 10)
//' time2 <- seq(0, 990, 10)
//' l <- 1
//' m <- 1
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' ## Get CI for Covaraince function
//' CI <- FCovCI(dir, n_files, time1, time2, basis_degree,
//'              boundary_knots, internal_knots, l, m)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' time1 <- seq(0, 990, 10)
//' time2 <- seq(0, 990, 10)
//' l <- 1
//' m <- 1
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for Covaraince function
//' CI <- FCovCI(dir, n_files, time1, time2, basis_degree,
//'              boundary_knots, internal_knots, l, m, X = X)
//'
//'
//' @export
// [[Rcpp::export]]
Rcpp::List FCovCI(const std::string dir,
                  const int n_files,
                  const arma::vec time1,
                  const arma::vec time2,
                  const int basis_degree,
                  const arma::vec boundary_knots,
                  const arma::vec internal_knots,
                  const int l,
                  const int m,
                  const double alpha = 0.05,
                  bool rescale = true,
                  const bool simultaneous = false,
                  const double burnin_prop = 0.1,
                  Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue){
  Rcpp::List CI;
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  int n_MCMC = sigma_i.n_elem;

  if(X.isNull()){
    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }

    // Get Phi Paramters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    if(l <= 0){
      Rcpp::stop("'l' must be positive");
    }
    if(l > phi_i(0,0).n_rows){
      Rcpp::stop("'l' must be less than or equal to the number of clusters in the model");
    }
    if(m <= 0){
      Rcpp::stop("'m' must be positive");
    }
    if(m > phi_i(0,0).n_rows){
      Rcpp::stop("'m' must be less than or equal to the number of clusters in the model");
    }

    if(rescale == true){
      if(phi_i(0,0).n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }

    arma::field<arma::cube> phi_samp1(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp1(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp1((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    arma::field<arma::cube> phi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), 1);
    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      phi_samp(i,0) = phi_samp1(i + std::floor((n_MCMC * n_files) * burnin_prop), 0);
    }

    // Make spline basis 1
    splines2::BSpline bspline1;
    bspline1 = splines2::BSpline(time1, internal_knots, basis_degree,
                                 boundary_knots);
    // Get Basis matrix (time1 x Phi.n_cols)
    arma::mat bspline_mat1{bspline1.basis(true)};
    // Make B_obs
    arma::mat B1 = bspline_mat1;

    // Make spline basis 2
    splines2::BSpline bspline2;
    bspline2 = splines2::BSpline(time2, internal_knots, basis_degree,
                                 boundary_knots);
    // Get Basis matrix (time2 x Phi.n_cols)
    arma::mat bspline_mat2{bspline2.basis(true)};
    // Make B_obs
    arma::mat B2 = bspline_mat2;

    // Initialize placeholders
    arma::mat CI_Upper = arma::zeros(time1.n_elem, time2.n_elem);
    arma::mat CI_50 = arma::zeros(time1.n_elem, time2.n_elem);
    arma::mat CI_Lower = arma::zeros(time2.n_elem, time2.n_elem);
    arma::cube cov_samp = arma::zeros(time1.n_elem, time2.n_elem, std::ceil((n_MCMC * n_files) * (1 - burnin_prop)));

    if(simultaneous == false){
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and Phi
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int a = 0; a < Z_samp.n_rows; a++){
              ph(a) = Z_samp(a,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          for(int b = 0; b < phi_samp(j,0).n_slices; b++){
            phi_samp(j,0).slice(b) = transform_mat * phi_samp(j,0).slice(b);
          }
        }
      }
      for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
        for(int j = 0; j < phi_samp(i,0).n_slices; j++){
          cov_samp.slice(i) = cov_samp.slice(i) + (B1 * (phi_samp(i,0).slice(j).row(l-1)).t() *
            (B2 * phi_samp(i,0).slice(j).row(m-1).t()).t());
        }
      }

      arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
      arma::vec q = arma::zeros(3);

      arma::vec ph1 = arma::zeros(cov_samp.n_slices);
      for(int i = 0; i < time1.n_elem; i++){
        for(int j = 0; j < time2.n_elem; j++){
          ph1 = cov_samp(arma::span(i), arma::span(j), arma::span::all);
          q = arma::quantile(ph1, p);
          CI_Upper(i,j) = q(2);
          CI_50(i,j) = q(1);
          CI_Lower(i,j) = q(0);
        }
      }
    }else{
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and Phi
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int a = 0; a < Z_samp.n_rows; a++){
              ph(a) = Z_samp(a,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          for(int b = 0; b < phi_samp(j,0).n_slices; b++){
            phi_samp(j,0).slice(b) = transform_mat * phi_samp(j,0).slice(b);
          }
        }
      }
      for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
        for(int j = 0; j < phi_samp(i,0).n_slices; j++){
          cov_samp.slice(i) = cov_samp.slice(i) + (B1 * (phi_samp(i,0).slice(j).row(l-1)).t() *
            (B2 * (phi_samp(i,0).slice(j).row(m-1)).t()).t());
        }
      }

      arma::mat cov_mean = arma::mean(cov_samp, 2);
      arma::mat cov_sd = arma::zeros(time1.n_elem, time2.n_elem);
      arma::vec ph2 = arma::zeros(cov_samp.n_slices);
      for(int i = 0; i < time1.n_elem; i++){
        for(int j = 0; j < time2.n_elem; j++){
          ph2 = cov_samp(arma::span(i), arma::span(j), arma::span::all);
          cov_sd(i,j) = arma::stddev(ph2);
        }
      }

      arma::vec C = arma::zeros(cov_samp.n_slices);
      arma::mat ph1 = arma::zeros(time1.n_elem, time2.n_elem);
      for(int i = 0; i < cov_samp.n_slices; i++){
        for(int j = 0; j < time1.n_elem; j++){
          for(int k = 0; k < time2.n_elem; k++){
            ph1(j,k) = std::abs((cov_samp(j,k,i) - cov_mean(j,k)) / cov_sd(j,k));
          }
        }
        C(i) = ph1.max();
      }

      arma::vec p = {1- alpha};
      arma::vec q = arma::zeros(1);
      q = arma::quantile(C, p);

      for(int i = 0; i < time1.n_elem; i++){
        for(int j = 0; j < time2.n_elem; j++){
          CI_Lower(i,j) = cov_mean(i,j) - q(0) * cov_sd(i,j);
          CI_50(i,j) = cov_mean(i,j);
          CI_Upper(i,j) =  cov_mean(i,j) + q(0) * cov_sd(i,j);
        }
      }
    }
    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("cov_trace", cov_samp));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }

    // Get Phi Paramters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    if(l <= 0){
      Rcpp::stop("'l' must be positive");
    }
    if(l > phi_i(0,0).n_rows){
      Rcpp::stop("'l' must be less than or equal to the number of clusters in the model");
    }
    if(m <= 0){
      Rcpp::stop("'m' must be positive");
    }
    if(m > phi_i(0,0).n_rows){
      Rcpp::stop("'m' must be less than or equal to the number of clusters in the model");
    }

    if(rescale == true){
      if(phi_i(0,0).n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }
    arma::field<arma::cube> phi_samp1(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp1(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp1((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    arma::field<arma::cube> phi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), 1);
    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      phi_samp(i,0) = phi_samp1(i + std::floor((n_MCMC * n_files) * burnin_prop), 0);
    }

    // Read in Xi parameters
    arma::field<arma::cube> xi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), phi_samp(0,0).n_rows);
    arma::field<arma::cube> xi_samp1(n_MCMC * n_files, phi_samp(0,0).n_rows);
    arma::field<arma::cube> xi_i;
    xi_i.load(dir + "Xi0.txt");
    for(int k = 0; k < phi_samp(0,0).n_rows; k++){
      for(int i = 0; i < n_MCMC; i++){
        xi_samp1(i,k) = xi_i(i,k);
      }
    }

    for(int i = 1; i < n_files; i++){
      xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
      for(int k = 0; k < phi_samp(0,0).n_rows; k++){
        for(int j = 0; j < n_MCMC; j++){
          xi_samp1((i * n_MCMC) + j, k) = xi_i(j,k);
        }
      }
    }

    if(xi_samp1(0,0).n_cols != X1.n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      for(int k = 0; k < phi_samp(0,0).n_rows; k++){
        xi_samp(i,k) = xi_samp1(i + std::floor((n_MCMC * n_files) * burnin_prop), k);
      }
    }

    // Make spline basis 1
    splines2::BSpline bspline1;
    bspline1 = splines2::BSpline(time1, internal_knots, basis_degree,
                                 boundary_knots);
    // Get Basis matrix (time1 x Phi.n_cols)
    arma::mat bspline_mat1{bspline1.basis(true)};
    // Make B_obs
    arma::mat B1 = bspline_mat1;

    // Make spline basis 2
    splines2::BSpline bspline2;
    bspline2 = splines2::BSpline(time2, internal_knots, basis_degree,
                                 boundary_knots);
    // Get Basis matrix (time2 x Phi.n_cols)
    arma::mat bspline_mat2{bspline2.basis(true)};
    // Make B_obs
    arma::mat B2 = bspline_mat2;

    // Initialize placeholders
    arma::cube CI_Upper = arma::zeros(time1.n_elem, time2.n_elem, X1.n_rows);
    arma::cube CI_50 = arma::zeros(time1.n_elem, time2.n_elem, X1.n_rows);
    arma::cube CI_Lower = arma::zeros(time2.n_elem, time2.n_elem, X1.n_rows);
    arma::field<arma::cube> cov_samp(X1.n_rows, 1);
    for(int i = 0; i < X1.n_rows; i++){
      cov_samp(i, 0) = arma::zeros(time1.n_elem, time2.n_elem, std::ceil((n_MCMC * n_files) * (1 - burnin_prop)));
    }

    if(simultaneous == false){
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and Phi
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        arma::field<arma::cube> xi_ph(Z_samp.n_cols, 1);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int a = 0; a < Z_samp.n_rows; a++){
              ph(a) = Z_samp(a,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          for(int b = 0; b < phi_samp(j,0).n_slices; b++){
            phi_samp(j,0).slice(b) = transform_mat * phi_samp(j,0).slice(b);
          }

          // rescale xi parameters
          for(int k = 0; k < Z_samp.n_cols; k++){
            xi_ph(k,0) = xi_samp(j,k);
          }

          for(int k = 0; k < Z_samp.n_cols; k++){
            xi_samp(j,k) = xi_ph(0,0) * transform_mat(k,0);
            for(int b = 1; b < Z_samp.n_cols; b++){
              xi_samp(j,k) = xi_samp(j,k) + (xi_ph(b,0) * transform_mat(k,b));
            }
          }
        }
      }
      for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
        for(int j = 0; j < phi_samp(i,0).n_slices; j++){
          for(int b = 0; b < X1.n_rows; b++){
            cov_samp(b,0).slice(i) = cov_samp(b,0).slice(i) + (B1 * ((phi_samp(i,0).slice(j).row(l-1)).t()+ xi_samp(i,l-1).slice(j) * X1.row(b).t()) *
              (B2 * ((phi_samp(i,0).slice(j).row(m-1)).t()+ xi_samp(i,m-1).slice(j) * X1.row(b).t())).t());
          }
        }
      }

      arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
      arma::vec q = arma::zeros(3);

      arma::vec ph1 = arma::zeros(cov_samp.n_slices);
      for(int b = 0; b < X1.n_rows; b++){
        for(int i = 0; i < time1.n_elem; i++){
          for(int j = 0; j < time2.n_elem; j++){
            ph1 = cov_samp(b,0)(arma::span(i), arma::span(j), arma::span::all);
            q = arma::quantile(ph1, p);
            CI_Upper(i,j,b) = q(2);
            CI_50(i,j,b) = q(1);
            CI_Lower(i,j,b) = q(0);
          }
        }
      }

    }else{
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and Phi
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        arma::field<arma::cube> xi_ph(Z_samp.n_cols, 1);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int a = 0; a < Z_samp.n_rows; a++){
              ph(a) = Z_samp(a,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          for(int b = 0; b < phi_samp(j,0).n_slices; b++){
            phi_samp(j,0).slice(b) = transform_mat * phi_samp(j,0).slice(b);
          }

          // rescale xi parameters
          for(int k = 0; k < Z_samp.n_cols; k++){
            xi_ph(k,0) = xi_samp(j,k);
          }

          for(int k = 0; k < Z_samp.n_cols; k++){
            xi_samp(j,k) = xi_ph(0,0) * transform_mat(k,0);
            for(int b = 1; b < Z_samp.n_cols; b++){
              xi_samp(j,k) = xi_samp(j,k) + (xi_ph(b,0) * transform_mat(k,b));
            }
          }

        }
      }
      for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
        for(int j = 0; j < phi_samp(i,0).n_slices; j++){
          for(int b = 0; b < X1.n_rows; b++){
            cov_samp(b,0).slice(i) = cov_samp(b,0).slice(i) + (B1 * ((phi_samp(i,0).slice(j).row(l-1)).t()+ xi_samp(i,l-1).slice(j) * X1.row(b).t()) *
              (B2 * ((phi_samp(i,0).slice(j).row(m-1)).t()+ xi_samp(i,m-1).slice(j) * X1.row(b).t())).t());
          }
        }
      }

      for(int b = 0; b < X1.n_rows; b++){
        arma::mat cov_mean = arma::mean(cov_samp(b,0), 2);
        arma::mat cov_sd = arma::zeros(time1.n_elem, time2.n_elem);
        arma::vec ph2 = arma::zeros(cov_samp.n_slices);
        for(int i = 0; i < time1.n_elem; i++){
          for(int j = 0; j < time2.n_elem; j++){
            ph2 = cov_samp(b,0)(arma::span(i), arma::span(j), arma::span::all);
            cov_sd(i,j) = arma::stddev(ph2);
          }
        }

        arma::vec C = arma::zeros(cov_samp.n_slices);
        arma::mat ph1 = arma::zeros(time1.n_elem, time2.n_elem);
        for(int i = 0; i < cov_samp.n_slices; i++){
          for(int j = 0; j < time1.n_elem; j++){
            for(int k = 0; k < time2.n_elem; k++){
              ph1(j,k) = std::abs((cov_samp(b,0)(j,k,i) - cov_mean(j,k)) / cov_sd(j,k));
            }
          }
          C(i) = ph1.max();
        }

        arma::vec p = {1- alpha};
        arma::vec q = arma::zeros(1);
        q = arma::quantile(C, p);

        for(int i = 0; i < time1.n_elem; i++){
          for(int j = 0; j < time2.n_elem; j++){
            CI_Lower(i,j,b) = cov_mean(i,j) - q(0) * cov_sd(i,j);
            CI_50(i,j,b) = cov_mean(i,j);
            CI_Upper(i,j,b) =  cov_mean(i,j) + q(0) * cov_sd(i,j);
          }
        }
      }

    }
    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("cov_trace", cov_samp));

  }

  return(CI);
}

//' Calculates the credible interval for the covariance (High Dimensional Functional Data)
//'
//' This function calculates a credible interval for the covariance function
//' between the l-th and m-th clusters, with the user specified coverage. This
//' function can handle covariate adjusted models, where the mean and covariance
//' functions depend on the covariates of interest. If not covariate adjusted, or
//' if the covariates only influence the mean structure, DO NOT specify \code{X} in
//' this function.
//' In order to run this function, the directory of the posterior samples needs
//' to be specified. The function will return the credible intervals and the median
//' posterior estimate of the covariance function at the time points specified by the
//' user (\code{time} variable). The user can specify if they would like the algorithm
//' to automatically rescale the parameters for interpretability (suggested). If
//' the user chooses to rescale, then all class memberships will be rescaled so
//' that at least one observation is in only one class. The user can also specify
//' if they want pointwise credible intervals or simultaneous credible intervals.
//' The simultaneous intervals will likely be wider than the pointwise credible
//' intervals.
//'
//' @name HDFCovCI
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param time1 Vector containing time points of interest for first cluster
//' @param time2 Vector containing time points of interest for second cluster
//' @param basis_degree Int containing the degree of B-splines used
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param l Int containing the 1st cluster group of which you want to get the credible interval for
//' @param m Int containing the 2nd cluster group of which you want to get the credible interval for
//' @param alpha Double specifying the percentile of the credible interval ((1 - alpha) * 100 percent)
//' @param rescale Boolean indicating whether or not we should rescale the Z variables so that there is at least one observation almost completely in one group
//' @param simultaneous Boolean indicating whether or not the credible intervals should be simultaneous credible intervals or pointwise credible intervals
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix containing covariates at points of interest (of dimension W x D (number of points of interest x number of covariates))
//' @return CI list containing the credible interval for the covariance function, as well as the median posterior estimate of the covariance function. Posterior estimates of the covariance function are also returned.
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{n_MCMC}}{must be an integer larger than or equal to 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{l}}{must be an integer larger than 1 and less than or equal to the number of clusters in the model}
//'   \item{\code{m}}{must be an integer larger than 1 and less than or equal to the number of clusters in the model}
//'   \item{\code{alpha}}{must be between 0 and 1}
//'   \item{\code{burnin_prop}}{must be less than 1 and greater than or equal to 0}
//'   \item{\code{X}}{must have the same number of columns as covariates in the model (D)}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "HDFunctional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' n_MCMC <- 200
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//' time <- time[[1]]
//' time1 <- time
//' time2 <- time
//' l <- 1
//' m <- 1
//' basis_degree <- c(2,2)
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' ## Get CI for Covaraince function
//' CI <- HDFCovCI(dir, n_files, time1, time2, basis_degree,
//'              boundary_knots, internal_knots, l, m)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "HDFunctional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' n_MCMC <- 200
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//' time <- time[[1]]
//' time1 <- time
//' time2 <- time
//' l <- 1
//' m <- 1
//' basis_degree <- c(2,2)
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//'
//' ## Get CI for Covaraince function
//' CI <- HDFCovCI(dir, n_files, time1, time2, basis_degree,
//'              boundary_knots, internal_knots, l, m)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "HDFunctional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' n_MCMC <- 200
//' time <- readRDS(system.file("test-data", "HDtime.RDS", package = "BayesFMMM"))
//' time <- time[[1]]
//' time1 <- time
//' time2 <- time
//' l <- 1
//' m <- 1
//' basis_degree <- c(2,2)
//' boundary_knots <- matrix(c(0, 0, 990, 990), nrow = 2)
//' internal_knots <- rep(list(c(250, 500, 750)), 2)
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for Covaraince function
//' CI <- HDFCovCI(dir, n_files, time1, time2, basis_degree,
//'              boundary_knots, internal_knots, l, m, X = X)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List HDFCovCI(const std::string dir,
                    const int n_files,
                    const arma::mat time1,
                    const arma::mat time2,
                    const arma::vec basis_degree,
                    const arma::mat boundary_knots,
                    const arma::field<arma::vec> internal_knots,
                    const int l,
                    const int m,
                    const double alpha = 0.05,
                    bool rescale = true,
                    const bool simultaneous = false,
                    const double burnin_prop = 0.1,
                    Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue){
  Rcpp::List CI;
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  int n_MCMC = sigma_i.n_elem;
  if(X.isNull()){
    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
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

    // Get Phi Paramters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    if(l <= 0){
      Rcpp::stop("'l' must be positive");
    }
    if(l > phi_i(0,0).n_rows){
      Rcpp::stop("'l' must be less than or equal to the number of clusters in the model");
    }
    if(m <= 0){
      Rcpp::stop("'m' must be positive");
    }
    if(m > phi_i(0,0).n_rows){
      Rcpp::stop("'m' must be less than or equal to the number of clusters in the model");
    }

    if(rescale == true){
      if(phi_i(0,0).n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }



    arma::field<arma::cube> phi_samp1(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp1(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp1((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    arma::field<arma::cube> phi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), 1);
    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      phi_samp(i,0) = phi_samp1(i + std::floor((n_MCMC * n_files) * (burnin_prop)), 0);
    }

    // Make spline basis 1
    arma::field<arma::mat> time1field(1,1);
    time1field(0,0) = time1;
    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time1field, 1, basis_degree,
                                                            boundary_knots, internal_knots);
    arma::mat B1 = B_obs(0,0);

    // Make spline basis 2
    arma::field<arma::mat> time2field(1,1);
    time2field(0,0) = time2;
    arma::field<arma::mat> B_obs2 = BayesFMMM::TensorBSpline(time1field, 1, basis_degree,
                                                             boundary_knots, internal_knots);
    arma::mat B2 = B_obs2(0,0);

    // Initialize placeholders
    arma::mat CI_Upper = arma::zeros(time1.n_rows, time2.n_rows);
    arma::mat CI_50 = arma::zeros(time1.n_rows, time2.n_rows);
    arma::mat CI_Lower = arma::zeros(time2.n_rows, time2.n_rows);
    arma::cube cov_samp = arma::zeros(time1.n_rows, time2.n_rows, std::ceil((n_MCMC * n_files) * (1 - burnin_prop)));

    if(simultaneous == false){
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and Phi
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int a = 0; a < Z_samp.n_rows; a++){
              ph(a) = Z_samp(a,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          for(int b = 0; b < phi_samp(j,0).n_slices; b++){
            phi_samp(j,0).slice(b) = transform_mat * phi_samp(j,0).slice(b);
          }
        }
      }
      for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
        for(int j = 0; j < phi_samp(i,0).n_slices; j++){
          cov_samp.slice(i) = cov_samp.slice(i) + (B1 * (phi_samp(i,0).slice(j).row(l-1)).t() *
            (B2 * phi_samp(i,0).slice(j).row(m-1).t()).t());
        }
      }

      arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
      arma::vec q = arma::zeros(3);

      arma::vec ph1 = arma::zeros(cov_samp.n_slices);
      for(int i = 0; i < time1.n_rows; i++){
        for(int j = 0; j < time2.n_rows; j++){
          ph1 = cov_samp(arma::span(i), arma::span(j), arma::span::all);
          q = arma::quantile(ph1, p);
          CI_Upper(i,j) = q(2);
          CI_50(i,j) = q(1);
          CI_Lower(i,j) = q(0);
        }
      }
    }else{
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and Phi
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int a = 0; a < Z_samp.n_rows; a++){
              ph(a) = Z_samp(a,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          for(int b = 0; b < phi_samp(j,0).n_slices; b++){
            phi_samp(j,0).slice(b) = transform_mat * phi_samp(j,0).slice(b);
          }
        }
      }
      for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
        for(int j = 0; j < phi_samp(i,0).n_slices; j++){
          cov_samp.slice(i) = cov_samp.slice(i) + (B1 * (phi_samp(i,0).slice(j).row(l-1)).t() *
            (B2 * (phi_samp(i,0).slice(j).row(m-1)).t()).t());
        }
      }

      arma::mat cov_mean = arma::mean(cov_samp, 2);
      arma::mat cov_sd = arma::zeros(time1.n_rows, time2.n_rows);
      arma::vec ph2 = arma::zeros(cov_samp.n_slices);
      for(int i = 0; i < time1.n_rows; i++){
        for(int j = 0; j < time2.n_rows; j++){
          ph2 = cov_samp(arma::span(i), arma::span(j), arma::span::all);
          cov_sd(i,j) = arma::stddev(ph2);
        }
      }

      arma::vec C = arma::zeros(cov_samp.n_slices);
      arma::mat ph1 = arma::zeros(time1.n_rows, time2.n_rows);
      for(int i = 0; i < cov_samp.n_slices; i++){
        for(int j = 0; j < time1.n_rows; j++){
          for(int k = 0; k < time2.n_rows; k++){
            ph1(j,k) = std::abs((cov_samp(j,k,i) - cov_mean(j,k)) / cov_sd(j,k));
          }
        }
        C(i) = ph1.max();
      }

      arma::vec p = {1- alpha};
      arma::vec q = arma::zeros(1);
      q = arma::quantile(C, p);

      for(int i = 0; i < time1.n_rows; i++){
        for(int j = 0; j < time2.n_rows; j++){
          CI_Lower(i,j) = cov_mean(i,j) - q(0) * cov_sd(i,j);
          CI_50(i,j) = cov_mean(i,j);
          CI_Upper(i,j) =  cov_mean(i,j) + q(0) * cov_sd(i,j);
        }
      }
    }
    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("cov_trace", cov_samp));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    for(int i = 0; i < basis_degree.n_elem; i++){
      if(basis_degree(i) <  1){
        Rcpp::stop("'basis_degree' elements must be an integer greater than or equal to 1");
      }
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

    // Get Phi Paramters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    if(l <= 0){
      Rcpp::stop("'l' must be positive");
    }
    if(l > phi_i(0,0).n_rows){
      Rcpp::stop("'l' must be less than or equal to the number of clusters in the model");
    }
    if(m <= 0){
      Rcpp::stop("'m' must be positive");
    }
    if(m > phi_i(0,0).n_rows){
      Rcpp::stop("'m' must be less than or equal to the number of clusters in the model");
    }

    if(rescale == true){
      if(phi_i(0,0).n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }

    arma::field<arma::cube> phi_samp1(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp1(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp1((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    arma::field<arma::cube> phi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), 1);
    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      phi_samp(i,0) = phi_samp1(i + std::floor((n_MCMC * n_files) * burnin_prop), 0);
    }

    // Read in Xi parameters
    arma::field<arma::cube> xi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), phi_samp(0,0).n_rows);
    arma::field<arma::cube> xi_samp1(n_MCMC * n_files, phi_samp(0,0).n_rows);
    arma::field<arma::cube> xi_i;
    xi_i.load(dir + "Xi0.txt");
    for(int k = 0; k < phi_samp(0,0).n_rows; k++){
      for(int i = 0; i < n_MCMC; i++){
        xi_samp1(i,k) = xi_i(i,k);
      }
    }

    for(int i = 1; i < n_files; i++){
      xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
      for(int k = 0; k < phi_samp(0,0).n_rows; k++){
        for(int j = 0; j < n_MCMC; j++){
          xi_samp1((i * n_MCMC) + j, k) = xi_i(j,k);
        }
      }
    }

    if(xi_samp1(0,0).n_cols != X1.n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      for(int k = 0; k < phi_samp(0,0).n_rows; k++){
        xi_samp(i,k) = xi_samp1(i + std::floor((n_MCMC * n_files) * burnin_prop), k);
      }
    }

    // Make spline basis 1
    arma::field<arma::mat> time1field(1,1);
    time1field(0,0) = time1;
    arma::field<arma::mat> B_obs = BayesFMMM::TensorBSpline(time1field, 1, basis_degree,
                                                            boundary_knots, internal_knots);
    arma::mat B1 = B_obs(0,0);

    // Make spline basis 2
    arma::field<arma::mat> time2field(1,1);
    time2field(0,0) = time2;
    arma::field<arma::mat> B_obs2 = BayesFMMM::TensorBSpline(time1field, 1, basis_degree,
                                                             boundary_knots, internal_knots);
    arma::mat B2 = B_obs2(0,0);

    // Initialize placeholders
    arma::cube CI_Upper = arma::zeros(time1.n_rows, time2.n_rows, X1.n_rows);
    arma::cube CI_50 = arma::zeros(time1.n_rows, time2.n_rows, X1.n_rows);
    arma::cube CI_Lower = arma::zeros(time2.n_rows, time2.n_rows, X1.n_rows);
    arma::field<arma::cube> cov_samp(X1.n_rows, 1);
    for(int i = 0; i < X1.n_rows; i++){
      cov_samp(i, 0) = arma::zeros(time1.n_rows, time2.n_rows, std::ceil((n_MCMC * n_files) * (1 - burnin_prop)));
    }

    if(simultaneous == false){
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and Phi
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        arma::field<arma::cube> xi_ph(Z_samp.n_cols, 1);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int a = 0; a < Z_samp.n_rows; a++){
              ph(a) = Z_samp(a,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          for(int b = 0; b < phi_samp(j,0).n_slices; b++){
            phi_samp(j,0).slice(b) = transform_mat * phi_samp(j,0).slice(b);
          }

          // rescale xi parameters
          for(int k = 0; k < Z_samp.n_cols; k++){
            xi_ph(k,0) = xi_samp(j,k);
          }

          for(int k = 0; k < Z_samp.n_cols; k++){
            xi_samp(j,k) = xi_ph(0,0) * transform_mat(k,0);
            for(int b = 1; b < Z_samp.n_cols; b++){
              xi_samp(j,k) = xi_samp(j,k) + (xi_ph(b,0) * transform_mat(k,b));
            }
          }
        }
      }
      for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
        for(int j = 0; j < phi_samp(i,0).n_slices; j++){
          for(int b = 0; b < X1.n_rows; b++){
            cov_samp(b,0).slice(i) = cov_samp(b,0).slice(i) + (B1 * ((phi_samp(i,0).slice(j).row(l-1)).t()+ xi_samp(i,l-1).slice(j) * X1.row(b).t()) *
              (B2 * ((phi_samp(i,0).slice(j).row(m-1)).t()+ xi_samp(i,m-1).slice(j) * X1.row(b).t())).t());
          }
        }
      }

      arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
      arma::vec q = arma::zeros(3);

      arma::vec ph1 = arma::zeros(cov_samp.n_slices);
      for(int b = 0; b < X1.n_rows; b++){
        for(int i = 0; i < time1.n_rows; i++){
          for(int j = 0; j < time2.n_rows; j++){
            ph1 = cov_samp(b,0)(arma::span(i), arma::span(j), arma::span::all);
            q = arma::quantile(ph1, p);
            CI_Upper(i,j,b) = q(2);
            CI_50(i,j,b) = q(1);
            CI_Lower(i,j,b) = q(0);
          }
        }
      }

    }else{
      if(rescale == true){
        // Get Z matrix
        arma::cube Z_i;
        Z_i.load(dir + "Z0.txt");
        arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
        Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
        for(int i = 1; i < n_files; i++){
          Z_i.load(dir + "Z" + std::to_string(i) +".txt");
          Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
        }

        arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                        std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
        Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                                 Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);
        // rescale Z and Phi
        arma::mat transform_mat;
        arma::vec ph = arma::zeros(Z_samp.n_rows);
        arma::field<arma::cube> xi_ph(Z_samp.n_cols, 1);
        for(int j = 0; j < Z_samp.n_slices; j++){
          transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
          int max_ind = 0;
          for(int i = 0; i < Z_samp.n_cols; i++){
            for(int a = 0; a < Z_samp.n_rows; a++){
              ph(a) = Z_samp(a,i,j);
            }
            max_ind = arma::index_max(ph);
            transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
          }
          for(int b = 0; b < phi_samp(j,0).n_slices; b++){
            phi_samp(j,0).slice(b) = transform_mat * phi_samp(j,0).slice(b);
          }

          // rescale xi parameters
          for(int k = 0; k < Z_samp.n_cols; k++){
            xi_ph(k,0) = xi_samp(j,k);
          }

          for(int k = 0; k < Z_samp.n_cols; k++){
            xi_samp(j,k) = xi_ph(0,0) * transform_mat(k,0);
            for(int b = 1; b < Z_samp.n_cols; b++){
              xi_samp(j,k) = xi_samp(j,k) + (xi_ph(b,0) * transform_mat(k,b));
            }
          }

        }
      }
      for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
        for(int j = 0; j < phi_samp(i,0).n_slices; j++){
          for(int b = 0; b < X1.n_rows; b++){
            cov_samp(b,0).slice(i) = cov_samp(b,0).slice(i) + (B1 * ((phi_samp(i,0).slice(j).row(l-1)).t()+ xi_samp(i,l-1).slice(j) * X1.row(b).t()) *
              (B2 * ((phi_samp(i,0).slice(j).row(m-1)).t()+ xi_samp(i,m-1).slice(j) * X1.row(b).t())).t());
          }
        }
      }

      for(int b = 0; b < X1.n_rows; b++){
        arma::mat cov_mean = arma::mean(cov_samp(b,0), 2);
        arma::mat cov_sd = arma::zeros(time1.n_rows, time2.n_rows);
        arma::vec ph2 = arma::zeros(cov_samp.n_slices);
        for(int i = 0; i < time1.n_rows; i++){
          for(int j = 0; j < time2.n_rows; j++){
            ph2 = cov_samp(b,0)(arma::span(i), arma::span(j), arma::span::all);
            cov_sd(i,j) = arma::stddev(ph2);
          }
        }

        arma::vec C = arma::zeros(cov_samp.n_slices);
        arma::mat ph1 = arma::zeros(time1.n_elem, time2.n_elem);
        for(int i = 0; i < cov_samp.n_slices; i++){
          for(int j = 0; j < time1.n_rows; j++){
            for(int k = 0; k < time2.n_rows; k++){
              ph1(j,k) = std::abs((cov_samp(b,0)(j,k,i) - cov_mean(j,k)) / cov_sd(j,k));
            }
          }
          C(i) = ph1.max();
        }

        arma::vec p = {1- alpha};
        arma::vec q = arma::zeros(1);
        q = arma::quantile(C, p);

        for(int i = 0; i < time1.n_rows; i++){
          for(int j = 0; j < time2.n_rows; j++){
            CI_Lower(i,j,b) = cov_mean(i,j) - q(0) * cov_sd(i,j);
            CI_50(i,j,b) = cov_mean(i,j);
            CI_Upper(i,j,b) =  cov_mean(i,j) + q(0) * cov_sd(i,j);
          }
        }
      }

    }
    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("cov_trace", cov_samp));

  }

  return(CI);
}

//' Calculates the credible interval for the covariance (Multivariate Data)
//'
//' This function calculates a credible interval for the covariance matrix
//' between the l-th and m-th clusters, with the user specified coverage. This
//' function can handle covariate adjusted models, where the mean and covariance
//' functions depend on the covariates of interest. If not covariate adjusted, or
//' if the covariates only influence the mean structure, DO NOT specify \code{X} in
//' this function.
//' In order to run this function, the directory of the posterior samples needs
//' to be specified. The function will return the credible intervals and the median
//' posterior estimate of the mean. The user can specify if they would like the
//' algorithm to automatically rescale the parameters for interpretability
//' (suggested). If the user chooses to rescale, then all class memberships will
//' be rescaled so that at least one observation is in only one class.
//'
//' @name MVCovCI
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param l Int containing the 1st cluster group of which you want to get the credible interval for
//' @param m Int containing the 2nd cluster group of which you want to get the credible interval for
//' @param alpha Double specifying the percentile of the credible interval ((1 - alpha) * 100 percent)
//' @param rescale Boolean indicating whether or not we should rescale the Z variables so that there is at least one observation almost completely in one group
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix containing covariates at points of interest (of dimension W x D (number of points of interest x number of covariates))
//' @return CI list containing the credible interval for the mean function, as well as the median posterior estimate of the mean function. Posterior estimates of the covariance function are also returned.
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{n_MCMC}}{must be an integer larger than or equal to 1}
//'   \item{\code{l}}{must be an integer larger than 1 and less than or equal to the number of clusters in the model}
//'   \item{\code{m}}{must be an integer larger than 1 and less than or equal to the number of clusters in the model}
//'   \item{\code{alpha}}{must be between 0 and 1}
//'   \item{\code{burnin_prop}}{must be less than 1 and greater than or equal to 0}
//'   \item{\code{X}}{must have the same number of columns as covariates in the model (D)}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Set Hyperparameters
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' l <- 1
//' m <- 1
//'
//' ## Get CI for cov function
//' CI <- MVCovCI(dir, n_files, l, m)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' l <- 1
//' m <- 1
//'
//' ## Get CI for cov function
//' CI <- MVCovCI(dir, n_files, l, m)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' l <- 1
//' m <- 1
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for cov function
//' CI <- MVCovCI(dir, n_files, l, m, X = X)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List MVCovCI(const std::string dir,
                   const int n_files,
                   const int l,
                   const int m,
                   const double alpha = 0.05,
                   bool rescale = true,
                   const double burnin_prop = 0.1,
                   Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue){
  Rcpp::List CI;
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  int n_MCMC = sigma_i.n_elem;

  if(X.isNull()){
    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    // Get Phi Paramters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    if(l <= 0){
      Rcpp::stop("'l' must be positive");
    }
    if(l > phi_i(0,0).n_rows){
      Rcpp::stop("'l' must be less than or equal to the number of clusters in the model");
    }
    if(m <= 0){
      Rcpp::stop("'m' must be positive");
    }
    if(m > phi_i(0,0).n_rows){
      Rcpp::stop("'m' must be less than or equal to the number of clusters in the model");
    }
    if(rescale == true){
      if(phi_i(0,0).n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }
    arma::field<arma::cube> phi_samp1(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp1(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp1((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    arma::field<arma::cube> phi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), 1);
    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      phi_samp(i,0) = phi_samp1(i + std::floor((n_MCMC * n_files) * (burnin_prop)), 0);
    }

    // Initialize placeholders
    arma::mat CI_Upper = arma::zeros(phi_i(0,0).n_cols, phi_i(0,0).n_cols);
    arma::mat CI_50 = arma::zeros(phi_i(0,0).n_cols, phi_i(0,0).n_cols);
    arma::mat CI_Lower = arma::zeros(phi_i(0,0).n_cols, phi_i(0,0).n_cols);
    arma::cube cov_samp = arma::zeros(phi_i(0,0).n_cols, phi_i(0,0).n_cols, std::ceil((n_MCMC * n_files) * (1 - burnin_prop)));

    if(rescale == true){
      // Get Z matrix
      arma::cube Z_i;
      Z_i.load(dir + "Z0.txt");
      arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
      Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
      for(int i = 1; i < n_files; i++){
        Z_i.load(dir + "Z" + std::to_string(i) +".txt");
        Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
      }
      arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                      std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
      Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                               Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);

      // rescale Z and nu
      arma::mat transform_mat;
      arma::vec ph = arma::zeros(Z_samp.n_rows);
      for(int j = 0; j < Z_samp.n_slices; j++){
        transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
        int max_ind = 0;
        for(int i = 0; i < Z_samp.n_cols; i++){
          for(int k = 0; k < Z_samp.n_rows; k++){
            ph(k) = Z_samp(k,i,j);
          }
          max_ind = arma::index_max(ph);
          transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
        }
        for(int i = 0; i < phi_samp(j,0).n_slices; i++){
          phi_samp(j,0).slice(i) = transform_mat * phi_samp(j,0).slice(i);
        }
      }

    }

    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      for(int j = 0; j < phi_samp(0,0).n_slices; j++){
        cov_samp.slice(i) = cov_samp.slice(i) + ((phi_samp(i,0).slice(j).row(l-1)).t() *
          (phi_samp(i,0).slice(j).row(m-1)));
      }
    }

    arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
    arma::vec q = arma::zeros(3);

    arma::vec ph1 = arma::zeros(cov_samp.n_slices);
    for(int i = 0; i < phi_i(0,0).n_cols; i++){
      for(int j = 0; j < phi_i(0,0).n_cols; j++){
        ph1 = cov_samp(arma::span(i), arma::span(j), arma::span::all);
        q = arma::quantile(ph1, p);
        CI_Upper(i,j) = q(2);
        CI_50(i,j) = q(1);
        CI_Lower(i,j) = q(0);
      }
    }

    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("cov_trace", cov_samp));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);
    if(n_files <= 0){
      Rcpp::stop("'n_files' must be greater than 0");
    }
    if(alpha < 0){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }
    if(alpha >= 1){
      Rcpp::stop("'alpha' must be between 0 and 1");
    }

    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    // Get Phi Paramters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    if(l <= 0){
      Rcpp::stop("'l' must be positive");
    }
    if(l > phi_i(0,0).n_rows){
      Rcpp::stop("'l' must be less than or equal to the number of clusters in the model");
    }
    if(m <= 0){
      Rcpp::stop("'m' must be positive");
    }
    if(m > phi_i(0,0).n_rows){
      Rcpp::stop("'m' must be less than or equal to the number of clusters in the model");
    }
    if(rescale == true){
      if(phi_i(0,0).n_rows > 2){
        rescale = false;
        Rcpp::Rcout << "Rescale property cannot be used for K > 2";
      }
    }

    arma::field<arma::cube> phi_samp1(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp1(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp1((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    arma::field<arma::cube> phi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), 1);
    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      phi_samp(i,0) = phi_samp1(i + std::floor((n_MCMC * n_files) * burnin_prop), 0);
    }

    // Read in Xi parameters
    arma::field<arma::cube> xi_samp(std::ceil((n_MCMC * n_files) * (1 - burnin_prop)), phi_samp(0,0).n_rows);
    arma::field<arma::cube> xi_samp1(n_MCMC * n_files, phi_samp(0,0).n_rows);
    arma::field<arma::cube> xi_i;
    xi_i.load(dir + "Xi0.txt");
    for(int k = 0; k < phi_samp(0,0).n_rows; k++){
      for(int i = 0; i < n_MCMC; i++){
        xi_samp1(i,k) = xi_i(i,k);
      }
    }

    for(int i = 1; i < n_files; i++){
      xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
      for(int k = 0; k < phi_samp(0,0).n_rows; k++){
        for(int j = 0; j < n_MCMC; j++){
          xi_samp1((i * n_MCMC) + j, k) = xi_i(j,k);
        }
      }
    }

    if(xi_samp1(0,0).n_cols != X1.n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      for(int k = 0; k < phi_samp(0,0).n_rows; k++){
        xi_samp(i,k) = xi_samp1(i + std::floor((n_MCMC * n_files) * burnin_prop), k);
      }
    }

    arma::cube CI_Upper = arma::zeros(phi_i(0,0).n_cols, phi_i(0,0).n_cols, X1.n_rows);
    arma::cube CI_50 = arma::zeros(phi_i(0,0).n_cols, phi_i(0,0).n_cols, X1.n_rows);
    arma::cube CI_Lower = arma::zeros(phi_i(0,0).n_cols, phi_i(0,0).n_cols, X1.n_rows);
    arma::field<arma::cube> cov_samp(X1.n_rows, 1);
    for(int i = 0; i < X1.n_rows; i++){
      cov_samp(i,0) = arma::zeros(phi_i(0,0).n_cols, phi_i(0,0).n_cols, std::ceil((n_MCMC * n_files) * (1 - burnin_prop)));
    }

    if(rescale == true){
      // Get Z matrix
      arma::cube Z_i;
      Z_i.load(dir + "Z0.txt");
      arma::cube Z_samp1 = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
      Z_samp1.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
      for(int i = 1; i < n_files; i++){
        Z_i.load(dir + "Z" + std::to_string(i) +".txt");
        Z_samp1.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
      }
      arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols,
                                      std::ceil((Z_i.n_slices * n_files)* (1 - burnin_prop)));
      Z_samp = Z_samp1.subcube(0, 0, std::floor(Z_i.n_slices * n_files * burnin_prop),
                               Z_samp1.n_rows-1, Z_samp1.n_cols-1, Z_samp1.n_slices-1);

      // rescale Z and nu
      arma::mat transform_mat;
      arma::vec ph = arma::zeros(Z_samp.n_rows);
      arma::field<arma::cube> xi_ph(Z_samp.n_cols, 1);
      for(int j = 0; j < Z_samp.n_slices; j++){
        transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
        int max_ind = 0;
        for(int i = 0; i < Z_samp.n_cols; i++){
          for(int k = 0; k < Z_samp.n_rows; k++){
            ph(k) = Z_samp(k,i,j);
          }
          max_ind = arma::index_max(ph);
          transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
        }
        for(int i = 0; i < phi_samp(j,0).n_slices; i++){
          phi_samp(j,0).slice(i) = transform_mat * phi_samp(j,0).slice(i);
        }
        // rescale xi parameters
        for(int k = 0; k < Z_samp.n_cols; k++){
          xi_ph(k,0) = xi_samp(j,k);
        }

        for(int k = 0; k < Z_samp.n_cols; k++){
          xi_samp(j,k) = xi_ph(0,0) * transform_mat(k,0);
          for(int b = 1; b < Z_samp.n_cols; b++){
            xi_samp(j,k) = xi_samp(j,k) + (xi_ph(b,0) * transform_mat(k,b));
          }
        }
      }
    }

    for(int i = 0; i < std::ceil((n_MCMC * n_files) * (1 - burnin_prop)); i++){
      for(int j = 0; j < phi_samp(0,0).n_slices; j++){
        for(int b = 0; b < X1.n_rows; b++){
          cov_samp(b,0).slice(i) = cov_samp(b,0).slice(i) + (((phi_samp(i,0).slice(j).row(l-1)).t()+ xi_samp(i,l-1).slice(j) * X1.row(b).t()) *
            (((phi_samp(i,0).slice(j).row(m-1)).t()+ xi_samp(i,m-1).slice(j) * X1.row(b).t())).t());
        }
      }
    }

    arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
    arma::vec q = arma::zeros(3);

    arma::vec ph1 = arma::zeros(cov_samp.n_slices);
    for(int i = 0; i < phi_i(0,0).n_cols; i++){
      for(int j = 0; j < phi_i(0,0).n_cols; j++){
        for(int b = 0; b < X1.n_rows; b++){
          ph1 = cov_samp(b,0)(arma::span(i), arma::span(j), arma::span::all);
          q = arma::quantile(ph1, p);
          CI_Upper(i,j,b) = q(2);
          CI_50(i,j,b) = q(1);
          CI_Lower(i,j,b) = q(0);
        }
      }
    }
    CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                             Rcpp::Named("CI_50", CI_50),
                             Rcpp::Named("CI_Lower", CI_Lower),
                             Rcpp::Named("cov_trace", cov_samp));

  }

  return(CI);
}

//' Calculates the credible interval for sigma squared for all types of data
//'
//' @name SigmaCI
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Integer containing the number of MCMC files
//' @param alpha Double specifying the percentile of the credible interval ((1 - alpha) * 100 percent)
//' @returns CI list containing the credible interval for the covariance function, as well as the median posterior estimate of the covariance function
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{alpha}}{must be between 0 and 1}
//' }
//'
//' @examples
//' ## Set Hyperparameters
//' dir <- system.file("test-data","", package = "BayesFMMM")
//' n_files <- 1
//'
//' ## Get CI for Z
//' CI <- SigmaCI(dir, n_files)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List SigmaCI(const std::string dir,
                   const int n_files,
                   const double alpha = 0.05){
  if(n_files <= 0){
    Rcpp::stop("'n_files' must be greater than 0");
  }
  if(alpha < 0){
    Rcpp::stop("'alpha' must be between 0 and 1");
  }
  if(alpha >= 1){
    Rcpp::stop("'alpha' must be between 0 and 1");
  }

  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
  sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
  for(int i = 1; i < n_files; i++){
    sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
    sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
  }

  arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
  arma::vec q = arma::zeros(3);
  q = arma::quantile(sigma_samp, p);

  double CI_Upper = q(2);
  double CI_50 = q(1);
  double CI_Lower = q(1);

  Rcpp::List CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                                      Rcpp::Named("CI_50", CI_50),
                                      Rcpp::Named("CI_Lower", CI_Lower));

  return(CI);
}

//' Calculates the credible interval for membership parameters Z
//'
//' This function constructs credible intervals using the MCMC samples of the
//' parameters. This function will handle high dimensional functional data,
//' functional data, and multivariate data.
//'
//' @name ZCI
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Integer containing the number of files per parameter
//' @param alpha Double specifying the percentile of the credible interval ((1 - alpha) * 100 percent)
//' @param rescale Boolean indicating whether or not we should rescale the Z variables so that there is at least one observation almost completely in one group
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @return CI List containing the desired credible values
//' @export
// [[Rcpp::export]]
Rcpp::List ZCI(const std::string dir,
               const int n_files,
               const double alpha = 0.05,
               bool rescale = true,
               const double burnin_prop = 0.1){
  arma::cube Z_i;
  Z_i.load(dir + "Z0.txt");
  arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
  Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;

  if(rescale == true){
    if(Z_samp.n_cols > 2){
      rescale = false;
      Rcpp::Rcout << "Rescale property cannot be used for K > 2";
    }
  }

  for(int i = 1; i < n_files; i++){
    Z_i.load(dir + "Z" + std::to_string(i) +".txt");
    Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
  }
  if(rescale == true){
    arma::mat transform_mat;
    arma::vec ph = arma::zeros(Z_samp.n_rows);
    for(int j = 0; j < Z_samp.n_slices; j++){
      transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
      int max_ind = 0;
      for(int i = 0; i < Z_samp.n_cols; i++){
        for(int k = 0; k < Z_samp.n_rows; k++){
          ph(k) = Z_samp(k,i,j);
        }
        max_ind = arma::index_max(ph);
        transform_mat.row(i) = Z_samp.slice(j).row(max_ind);
      }
      //Z_samp.slice(j) =  Z_samp.slice(j) * arma::inv(transform_mat);
      Z_samp.slice(j) =  arma::solve(transform_mat.t(), Z_samp.slice(j).t(), arma::solve_opts::no_approx).t();
    }
  }

  arma::vec p = {alpha/2, 0.5, 1 - (alpha/2)};
  arma::vec q = arma::zeros(3);

  arma::mat CI_Upper = arma::zeros(Z_i.n_rows, Z_i.n_cols);
  arma::mat CI_50 = arma::zeros(Z_i.n_rows, Z_i.n_cols);
  arma::mat CI_Lower = arma::zeros(Z_i.n_rows, Z_i.n_cols);

  arma::vec ph(std::ceil(Z_samp.n_slices * (1 - burnin_prop)), arma::fill::zeros);
  for(int i = 0; i < Z_i.n_rows; i++){
    for(int j = 0; j < Z_i.n_cols; j++){
      for(int l = 0; l < std::ceil(Z_samp.n_slices * (1 - burnin_prop)); l++){
        ph(l) = Z_samp(i,j, l + std::floor(Z_samp.n_slices * burnin_prop));
      }
      q = arma::quantile(ph, p);
      CI_Upper(i,j) = q(2);
      CI_50(i,j) = q(1);
      CI_Lower(i,j) = q(0);
    }
  }

  Rcpp::List CI =  Rcpp::List::create(Rcpp::Named("CI_Upper", CI_Upper),
                                      Rcpp::Named("CI_50", CI_50),
                                      Rcpp::Named("CI_Lower", CI_Lower));

  return(CI);
}

//' Calculates the DIC of a functional model
//'
//' @name FDIC
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param basis_degree Int containing the degree of B-splines used
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param time Field of vectors containing time points at which the function was observed
//' @param Y Field of vectors containing observed values of the function
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param cov_adj Boolean containing whether or not the covariance structure depends on the covariates of interest
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//'
//' @returns DIC Double containing DIC value
//' @export
// [[Rcpp::export]]
double FDIC(const std::string dir,
            const int n_files,
            const int basis_degree,
            const arma::vec boundary_knots,
            const arma::vec internal_knots,
            const arma::field<arma::vec> time,
            const arma::field<arma::vec> Y,
            const double burnin_prop = 0.2,
            Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
            const bool cov_adj = false){
  double DIC;

  if(X.isNull()){
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    // Get sigma parameters
    arma::vec sigma_i;
    sigma_i.load(dir + "Sigma0.txt");
    int n_MCMC = sigma_i.n_elem;
    arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
    sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
    for(int i = 1; i < n_files; i++){
      sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
      sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
    }

    // Get Nu parameters
    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                      (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    // Get Phi parameters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    // Get Z parameters
    arma::cube Z_i;
    Z_i.load(dir + "Z0.txt");
    arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
    Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
    for(int i = 1; i < n_files; i++){
      Z_i.load(dir + "Z" + std::to_string(i) +".txt");
      Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
    }


    // Get chi parameters
    arma::cube chi_i;
    chi_i.load(dir + "Chi0.txt");
    arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
    chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
    for(int i = 1; i < n_files; i++){
      chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
      chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                       (chi_i.n_slices)*(i+1) - 1) = chi_i;
    }

    // Make spline basis
    arma::field<arma::mat> B_obs(Z_samp.n_rows, 1);
    splines2::BSpline bspline2;
    for(int i = 0; i < Z_samp.n_rows; i++){
      bspline2 = splines2::BSpline(time(i,0),  internal_knots, basis_degree,
                                   boundary_knots);
      // Get Basis matrix (time2 x Phi.n_cols)
      arma::mat bspline_mat2{bspline2.basis(true)};
      // Make B_obs
      B_obs(i,0) = bspline_mat2;
    }

    double expected_log_f = 0;
    for(int i = std::floor(burnin_prop *nu_samp.n_slices) ; i < nu_samp.n_slices; i++){
      expected_log_f = expected_log_f + BayesFMMM::calcLikelihood(Y, B_obs, nu_samp.slice(i),
                                                                  phi_samp(i,0), Z_samp.slice(i),
                                                                  chi_samp.slice(i), sigma_samp(i));
    }
    expected_log_f = expected_log_f / std::ceil((1-burnin_prop) *nu_samp.n_slices);

    double f_hat = 0;
    double f_hat_ij = 0;
    for(int i = 0; i < Z_samp.n_rows; i++){
      for(int j = 0; j < time(i,0).n_elem; j++){
        f_hat_ij = 0;
        for(int n = std::floor(burnin_prop *nu_samp.n_slices); n < nu_samp.n_slices; n++){
          f_hat_ij = f_hat_ij + BayesFMMM::calcDIC2(Y(i,0), B_obs(i,0), nu_samp.slice(n), phi_samp(n,0),
                                                    Z_samp.slice(n), chi_samp.slice(n), i, j,
                                                    sigma_samp(n));
        }
        f_hat = f_hat + std::log(f_hat_ij / std::ceil((1-burnin_prop) *nu_samp.n_slices));
      }
    }

    DIC = (2 * f_hat) - (4 * expected_log_f);
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    // Get sigma parameters
    arma::vec sigma_i;
    sigma_i.load(dir + "Sigma0.txt");
    int n_MCMC = sigma_i.n_elem;
    arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
    sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
    for(int i = 1; i < n_files; i++){
      sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
      sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
    }

    // Get Nu parameters
    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                      (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    // Get Phi parameters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    // Get Z parameters
    arma::cube Z_i;
    Z_i.load(dir + "Z0.txt");
    arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
    Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
    for(int i = 1; i < n_files; i++){
      Z_i.load(dir + "Z" + std::to_string(i) +".txt");
      Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
    }

    // Get chi parameters
    arma::cube chi_i;
    chi_i.load(dir + "Chi0.txt");
    arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
    chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
    for(int i = 1; i < n_files; i++){
      chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
      chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                       (chi_i.n_slices)*(i+1) - 1) = chi_i;
    }

    // Get eta parameters
    arma::field<arma::cube> eta_samp(n_MCMC * n_files, 1);
    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    for(int i = 0; i < n_MCMC; i++){
      eta_samp(i,0) = eta_i(i,0);
    }

    if(X1.n_cols != eta_samp(0,0).n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        eta_samp((i * n_MCMC) + j, 0) = eta_i(j,0);
      }
    }

    arma::field<arma::cube> xi_samp(n_MCMC * n_files, nu_samp.n_rows);
    if(cov_adj == false){
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int i = 0; i < n_MCMC * n_files; i ++){
          xi_samp(i,k) = arma::zeros(nu_samp.n_cols, X1.n_cols, phi_samp(0,0).n_slices);
        }
      }
    }else{
      arma::field<arma::cube> xi_i;
      xi_i.load(dir + "Xi0.txt");
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int i = 0; i < n_MCMC; i++){
          xi_samp(i,k) = xi_i(i,k);
        }
      }

      for(int i = 1; i < n_files; i++){
        xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
        for(int k = 0; k < nu_samp.n_rows; k++){
          for(int j = 0; j < n_MCMC; j++){
            xi_samp((i * n_MCMC) + j, k) = xi_i(j,k);
          }
        }
      }
    }

    // Make spline basis
    arma::field<arma::mat> B_obs(Z_samp.n_rows, 1);
    splines2::BSpline bspline2;
    for(int i = 0; i < Z_samp.n_rows; i++){
      bspline2 = splines2::BSpline(time(i,0),  internal_knots, basis_degree,
                                   boundary_knots);
      // Get Basis matrix (time2 x Phi.n_cols)
      arma::mat bspline_mat2{bspline2.basis(true)};
      // Make B_obs
      B_obs(i,0) = bspline_mat2;
    }

    double expected_log_f = 0;
    for(int i = std::floor(burnin_prop *nu_samp.n_slices) ; i < nu_samp.n_slices; i++){
      expected_log_f = expected_log_f + BayesFMMM::calcLikelihoodCovariateAdj(Y, B_obs, nu_samp.slice(i), eta_samp(i,0),
                                                                  phi_samp(i,0), xi_samp, Z_samp.slice(i),
                                                                  chi_samp.slice(i), i, X1, sigma_samp(i));
    }
    expected_log_f = expected_log_f / std::ceil((1-burnin_prop) * nu_samp.n_slices);

    double f_hat = 0;
    double f_hat_ij = 0;
    for(int i = 0; i < Z_samp.n_rows; i++){
      for(int j = 0; j < time(i,0).n_elem; j++){
        f_hat_ij = 0;
        for(int n = std::floor(burnin_prop *nu_samp.n_slices); n < nu_samp.n_slices; n++){
          f_hat_ij = f_hat_ij + BayesFMMM::calcDIC2CovariateAdj(Y(i,0), X1, B_obs(i,0), nu_samp.slice(n), eta_samp(n,0),
                                                                phi_samp(n,0), xi_samp, Z_samp.slice(n), chi_samp.slice(n),
                                                                n, i, j, sigma_samp(n));
        }
        f_hat = f_hat + std::log(f_hat_ij / std::ceil((1-burnin_prop) * nu_samp.n_slices));
      }
    }

    DIC = (2 * f_hat) - (4 * expected_log_f);

  }

  return(DIC);
}

//' Calculates the AIC of a functional model
//'
//' @name FAIC
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param basis_degree Int containing the degree of B-splines used
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param time Field of vectors containing time points at which the function was observed
//' @param Y Field of vectors containing observed values of the function
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param cov_adj Boolean containing whether or not the covariance structure depends on the covariates of interest
//'
//' @returns AIC Double containing AIC value
//' @export
// [[Rcpp::export]]
double FAIC(const std::string dir,
            const int n_files,
            const int basis_degree,
            const arma::vec boundary_knots,
            const arma::vec internal_knots,
            const arma::field<arma::vec> time,
            const arma::field<arma::vec> Y,
            const double burnin_prop = 0.2,
            Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
            const bool cov_adj = false){
  double AIC;

  if(X.isNull()){
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    // Get sigma parameters
    arma::vec sigma_i;
    sigma_i.load(dir + "Sigma0.txt");
    int n_MCMC = sigma_i.n_elem;
    arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
    sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
    for(int i = 1; i < n_files; i++){
      sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
      sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
    }

    // Get Nu parameters
    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                      (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    // Get Phi parameters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    // Get Z parameters
    arma::cube Z_i;
    Z_i.load(dir + "Z0.txt");
    arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
    Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
    for(int i = 1; i < n_files; i++){
      Z_i.load(dir + "Z" + std::to_string(i) +".txt");
      Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
    }

    // Get chi parameters
    arma::cube chi_i;
    chi_i.load(dir + "Chi0.txt");
    arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
    chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
    for(int i = 1; i < n_files; i++){
      chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
      chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                       (chi_i.n_slices)*(i+1) - 1) = chi_i;
    }

    // Make spline basis
    arma::field<arma::mat> B_obs(Z_samp.n_rows, 1);
    splines2::BSpline bspline2;
    for(int i = 0; i < Z_samp.n_rows; i++){
      bspline2 = splines2::BSpline(time(i,0),  internal_knots, basis_degree,
                                   boundary_knots);
      // Get Basis matrix (time2 x Phi.n_cols)
      arma::mat bspline_mat2{bspline2.basis(true)};
      // Make B_obs
      B_obs(i,0) = bspline_mat2;
    }

    // Estimate individual curve fit
    arma::field<arma::mat> curve_fit(Z_i.n_rows, 1);
    arma::rowvec Z_ph(Z_i.n_cols, arma::fill::zeros);
    for(int i = 0; i < Z_i.n_rows; i++){
      curve_fit(i,0) = arma::zeros(std::ceil((1 - burnin_prop) * sigma_i.n_elem * n_files), Y(i,0).n_elem);
      for(int j = std::ceil(burnin_prop * sigma_i.n_elem * n_files); j < sigma_i.n_elem * n_files; j++){
        Z_ph = Z_samp(arma::span(i), arma::span::all, arma::span(j));
        curve_fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) = Z_ph * nu_samp.slice(j) * B_obs(i,0).t();
        for(int k = 0; k < chi_samp.n_cols; k++){
          curve_fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) = curve_fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) + (chi_samp(i, k, j) *
            Z_ph * phi_samp(j,0).slice(k) * B_obs(i,0).t());
        }
      }
    }

    // Get mean curve fit
    arma::field<arma::rowvec> mean_curve_fit(Z_i.n_rows, 1);
    for(int i = 0; i < Z_i.n_rows; i++){
      mean_curve_fit(i,0) = arma::mean(curve_fit(i,0), 0);
    }

    // Get posterior mean of sigma^2
    double mean_sigma = arma::mean(sigma_samp);

    // Calculate Log likelihood
    double log_lik = 0;
    for(int i = 0; i < Z_samp.n_rows; i++){
      for(int j = 0; j < Y(i,0).n_elem; j++){
        log_lik = log_lik + R::dnorm(Y(i,0)(j), mean_curve_fit(i,0)(j),
                                     std::sqrt(mean_sigma), true);
      }
    }

    // Calculate AIC
    AIC =  2*((Z_samp.n_rows + phi_samp(0,0).n_cols) * Z_samp.n_cols +
      2 * phi_samp(0,0).n_cols * phi_samp(0,0).n_slices * phi_samp(0,0).n_rows +
      2 + 4 * Z_samp.n_cols + chi_samp.n_rows * chi_samp.n_cols + (phi_samp(0,0).n_slices*Z_samp.n_cols)) - (2 * log_lik);
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    // Get sigma parameters
    arma::vec sigma_i;
    sigma_i.load(dir + "Sigma0.txt");
    int n_MCMC = sigma_i.n_elem;
    arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
    sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
    for(int i = 1; i < n_files; i++){
      sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
      sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
    }

    // Get Nu parameters
    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                      (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    // Get Phi parameters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    // Get Z parameters
    arma::cube Z_i;
    Z_i.load(dir + "Z0.txt");
    arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
    Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
    for(int i = 1; i < n_files; i++){
      Z_i.load(dir + "Z" + std::to_string(i) +".txt");
      Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
    }

    // Get chi parameters
    arma::cube chi_i;
    chi_i.load(dir + "Chi0.txt");
    arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
    chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
    for(int i = 1; i < n_files; i++){
      chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
      chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                       (chi_i.n_slices)*(i+1) - 1) = chi_i;
    }

    // Get eta parameters
    arma::field<arma::cube> eta_samp(n_MCMC * n_files, 1);
    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    for(int i = 0; i < n_MCMC; i++){
      eta_samp(i,0) = eta_i(i,0);
    }

    if(X1.n_cols != eta_samp(0,0).n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        eta_samp((i * n_MCMC) + j, 0) = eta_i(j,0);
      }
    }

    arma::field<arma::cube> xi_samp(n_MCMC * n_files, nu_samp.n_rows);
    if(cov_adj == false){
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int i = 0; i < n_MCMC * n_files; i ++){
          xi_samp(i,k) = arma::zeros(nu_samp.n_cols, X1.n_cols, phi_samp(0,0).n_slices);
        }
      }
    }else{
      arma::field<arma::cube> xi_i;
      xi_i.load(dir + "Xi0.txt");
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int i = 0; i < n_MCMC; i++){
          xi_samp(i,k) = xi_i(i,k);
        }
      }

      for(int i = 1; i < n_files; i++){
        xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
        for(int k = 0; k < nu_samp.n_rows; k++){
          for(int j = 0; j < n_MCMC; j++){
            xi_samp((i * n_MCMC) + j, k) = xi_i(j,k);
          }
        }
      }
    }

    // Make spline basis
    arma::field<arma::mat> B_obs(Z_samp.n_rows, 1);
    splines2::BSpline bspline2;
    for(int i = 0; i < Z_samp.n_rows; i++){
      bspline2 = splines2::BSpline(time(i,0),  internal_knots, basis_degree,
                                   boundary_knots);
      // Get Basis matrix (time2 x Phi.n_cols)
      arma::mat bspline_mat2{bspline2.basis(true)};
      // Make B_obs
      B_obs(i,0) = bspline_mat2;
    }

    // Estimate individual curve fit
    arma::field<arma::mat> curve_fit(Z_i.n_rows, 1);
    for(int i = 0; i < Z_i.n_rows; i++){
      curve_fit(i,0) = arma::zeros(std::ceil((1 - burnin_prop) * sigma_i.n_elem * n_files), Y(i,0).n_elem);
      for(int j = std::ceil(burnin_prop * sigma_i.n_elem * n_files); j < sigma_i.n_elem * n_files; j++){
        for(int l = 0; l < Y(i,0).n_elem; l++){
          for(int k = 0; k < Z_samp.n_cols; k++){
            curve_fit(i,0)(j - std::floor(burnin_prop * sigma_i.n_elem * n_files), l) = curve_fit(i,0)(j - std::floor(burnin_prop * sigma_i.n_elem * n_files), l) + (Z_samp(i,k,j) * arma::dot(nu_samp.slice(j).row(k).t() +
              (eta_samp(j,0).slice(k) * X1.row(i).t()), B_obs(i,0).row(l)));
            for(int m = 0; m < chi_samp.n_cols; m++){
              curve_fit(i,0)(j - std::floor(burnin_prop * sigma_i.n_elem * n_files),l) = curve_fit(i,0)(j - std::floor(burnin_prop * sigma_i.n_elem * n_files), l) + (Z_samp(i,k,j) * chi_samp(i,m,j) * arma::dot(phi_samp(j,0).slice(m).row(k).t() +
                (xi_samp(j,k).slice(m) * X1.row(i).t()), B_obs(i,0).row(l)));
            }
          }
        }
      }
    }

    arma::field<arma::rowvec> mean_curve_fit(Z_samp.n_rows, 1);
    // Get mean curve fit
    for(int i = 0; i < Z_i.n_rows; i++){
      mean_curve_fit(i,0) = arma::mean(curve_fit(i,0), 0);
    }

    // Get posterior mean of sigma^2
    double mean_sigma = arma::mean(sigma_samp);

    // Calculate Log likelihood
    double log_lik = 0;
    for(int i = 0; i < Z_samp.n_rows; i++){
      for(int j = 0; j < Y(i,0).n_elem; j++){
        log_lik = log_lik + R::dnorm(Y(i,0)(j), mean_curve_fit(i,0)(j),
                                     std::sqrt(mean_sigma), true);
      }
    }

    // Calculate AIC
    AIC =  2*((Z_samp.n_rows + phi_samp(0,0).n_cols) * Z_samp.n_cols +
      2 * phi_samp(0,0).n_cols * phi_samp(0,0).n_slices * phi_samp(0,0).n_rows +
      2 + 4 * Z_samp.n_cols + chi_samp.n_rows * chi_samp.n_cols + (phi_samp(0,0).n_slices * Z_samp.n_cols) +
      eta_samp(0,0).n_rows * eta_samp(0,0).n_cols * eta_samp(0,0).n_slices +
      eta_samp(0,0).n_cols * eta_samp(0,0).n_slices) - (2 * log_lik);
    if(cov_adj == true){
      AIC = AIC + 2*(2 * eta_samp(0,0).n_rows * eta_samp(0,0).n_cols * eta_samp(0,0).n_slices * phi_samp(0,0).n_slices +
        eta_samp(0,0).n_cols * eta_samp(0,0).n_slices * phi_samp(0,0).n_slices + 2 * eta_samp(0,0).n_cols * eta_samp(0,0).n_slices);
    }

  }

  return(AIC);
}

//' Calculates the BIC of a functional model
//'
//' @name FBIC
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param basis_degree Int containing the degree of B-splines used
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param time Field of vectors containing time points at which the function was observed
//' @param Y Field of vectors containing observed values of the function
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix of covariates, where each row corresponds to an observation (if covariate adjusted)
//' @param cov_adj Boolean containing whether or not the covariance structure depends on the covariates of interest
//'
//' @returns BIC Double containing BIC value
//' @export
// [[Rcpp::export]]
double FBIC(const std::string dir,
            const int n_files,
            const int basis_degree,
            const arma::vec boundary_knots,
            const arma::vec internal_knots,
            const arma::field<arma::vec> time,
            const arma::field<arma::vec> Y,
            const double burnin_prop = 0.2,
            Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
            const bool cov_adj = false){
  double BIC;

  if(X.isNull()){
    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    // Get sigma parameters
    arma::vec sigma_i;
    sigma_i.load(dir + "Sigma0.txt");
    int n_MCMC = sigma_i.n_elem;
    arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
    sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
    for(int i = 1; i < n_files; i++){
      sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
      sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
    }

    // Get Nu parameters
    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                      (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    // Get Phi parameters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    // Get Z parameters
    arma::cube Z_i;
    Z_i.load(dir + "Z0.txt");
    arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
    Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
    for(int i = 1; i < n_files; i++){
      Z_i.load(dir + "Z" + std::to_string(i) +".txt");
      Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
    }

    // Get chi parameters
    arma::cube chi_i;
    chi_i.load(dir + "Chi0.txt");
    arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
    chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
    for(int i = 1; i < n_files; i++){
      chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
      chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                       (chi_i.n_slices)*(i+1) - 1) = chi_i;
    }

    // Make spline basis
    arma::field<arma::mat> B_obs(Z_samp.n_rows, 1);
    splines2::BSpline bspline2;
    for(int i = 0; i < Z_samp.n_rows; i++){
      bspline2 = splines2::BSpline(time(i,0),  internal_knots, basis_degree,
                                   boundary_knots);
      // Get Basis matrix (time2 x Phi.n_cols)
      arma::mat bspline_mat2{bspline2.basis(true)};
      // Make B_obs
      B_obs(i,0) = bspline_mat2;
    }

    // Estimate individual curve fit
    arma::field<arma::mat> curve_fit(Z_i.n_rows, 1);
    arma::rowvec Z_ph(Z_i.n_cols, arma::fill::zeros);
    for(int i = 0; i < Z_i.n_rows; i++){
      curve_fit(i,0) = arma::zeros(std::ceil((1 - burnin_prop) * sigma_i.n_elem * n_files), Y(i,0).n_elem);
      for(int j = std::floor(burnin_prop * sigma_i.n_elem * n_files); j < sigma_i.n_elem * n_files; j++){
        Z_ph = Z_samp(arma::span(i), arma::span::all, arma::span(j));
        curve_fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) = Z_ph * nu_samp.slice(j) * B_obs(i,0).t();
        for(int k = 0; k < chi_samp.n_cols; k++){
          curve_fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) = curve_fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) + (chi_samp(i, k, j) *
            Z_ph * phi_samp(j,0).slice(k) * B_obs(i,0).t());
        }
      }
    }

    // Get mean curve fit
    arma::field<arma::rowvec> mean_curve_fit(Z_i.n_rows, 1);
    for(int i = 0; i < Z_i.n_rows; i++){
      mean_curve_fit(i,0) = arma::mean(curve_fit(i,0), 0);
    }

    // Get posterior mean of sigma^2
    double mean_sigma = arma::mean(sigma_samp);

    // Calculate Log likelihood
    double log_lik = 0;
    for(int i = 0; i < Z_samp.n_rows; i++){
      for(int j = 0; j < Y(i,0).n_elem; j++){
        log_lik = log_lik + R::dnorm(Y(i,0)(j), mean_curve_fit(i,0)(j),
                                     std::sqrt(mean_sigma), true);
      }
    }
    double tilde_N = 0;
    for(int i = 0; i < Z_samp.n_rows; i++){
      tilde_N = tilde_N + Y(i,0).n_elem;
    }

    // Calculate BIC
    BIC = (2 * log_lik) - (std::log(tilde_N) *
      ((Z_samp.n_rows + phi_samp(0,0).n_cols) * Z_samp.n_cols +
      2 * phi_samp(0,0).n_cols * phi_samp(0,0).n_slices * phi_samp(0,0).n_rows +
      2 + 4 * Z_samp.n_cols + chi_samp.n_rows * chi_samp.n_cols + (phi_samp(0,0).n_slices*Z_samp.n_cols)));
  }else{
    Rcpp::NumericMatrix X_(X);
    arma::mat X1 = Rcpp::as<arma::mat>(X_);

    if(basis_degree <  1){
      Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
    }
    for(int i = 0; i < internal_knots.n_elem; i++){
      if(boundary_knots(0) >= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
      }
      if(boundary_knots(1) <= internal_knots(i)){
        Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
      }
    }
    if(burnin_prop < 0){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }
    if(burnin_prop >= 1){
      Rcpp::stop("'burnin_prop' must be between 0 and 1");
    }

    // Get sigma parameters
    arma::vec sigma_i;
    sigma_i.load(dir + "Sigma0.txt");
    int n_MCMC = sigma_i.n_elem;
    arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
    sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
    for(int i = 1; i < n_files; i++){
      sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
      sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
    }

    // Get Nu parameters
    arma::cube nu_i;
    nu_i.load(dir + "Nu0.txt");
    arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
    nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
    for(int i = 1; i < n_files; i++){
      nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
      nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                      (nu_i.n_slices)*(i+1) - 1) = nu_i;
    }

    // Get Phi parameters
    arma::field<arma::cube> phi_i;
    phi_i.load(dir + "Phi0.txt");
    arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
    for(int i = 0; i < n_MCMC; i++){
      phi_samp(i,0) = phi_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
      }
    }

    // Get Z parameters
    arma::cube Z_i;
    Z_i.load(dir + "Z0.txt");
    arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
    Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
    for(int i = 1; i < n_files; i++){
      Z_i.load(dir + "Z" + std::to_string(i) +".txt");
      Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
    }

    // Get chi parameters
    arma::cube chi_i;
    chi_i.load(dir + "Chi0.txt");
    arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
    chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
    for(int i = 1; i < n_files; i++){
      chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
      chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                       (chi_i.n_slices)*(i+1) - 1) = chi_i;
    }

    // Get eta parameters
    arma::field<arma::cube> eta_samp(n_MCMC * n_files, 1);
    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    for(int i = 0; i < n_MCMC; i++){
      eta_samp(i,0) = eta_i(i,0);
    }

    if(X1.n_cols != eta_samp(0,0).n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        eta_samp((i * n_MCMC) + j, 0) = eta_i(j,0);
      }
    }

    arma::field<arma::cube> xi_samp(n_MCMC * n_files, nu_samp.n_rows);
    if(cov_adj == false){
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int i = 0; i < n_MCMC * n_files; i ++){
          xi_samp(i,k) = arma::zeros(nu_samp.n_cols, X1.n_cols, phi_samp(0,0).n_slices);
        }
      }
    }else{
      arma::field<arma::cube> xi_i;
      xi_i.load(dir + "Xi0.txt");
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int i = 0; i < n_MCMC; i++){
          xi_samp(i,k) = xi_i(i,k);
        }
      }

      for(int i = 1; i < n_files; i++){
        xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
        for(int k = 0; k < nu_samp.n_rows; k++){
          for(int j = 0; j < n_MCMC; j++){
            xi_samp((i * n_MCMC) + j, k) = xi_i(j,k);
          }
        }
      }
    }

    // Make spline basis
    arma::field<arma::mat> B_obs(Z_samp.n_rows, 1);
    splines2::BSpline bspline2;
    for(int i = 0; i < Z_samp.n_rows; i++){
      bspline2 = splines2::BSpline(time(i,0),  internal_knots, basis_degree,
                                   boundary_knots);
      // Get Basis matrix (time2 x Phi.n_cols)
      arma::mat bspline_mat2{bspline2.basis(true)};
      // Make B_obs
      B_obs(i,0) = bspline_mat2;
    }

    // Estimate individual curve fit
    arma::field<arma::mat> curve_fit(Z_i.n_rows, 1);
    for(int i = 0; i < Z_i.n_rows; i++){
      curve_fit(i,0) = arma::zeros(std::ceil((1 - burnin_prop) * sigma_i.n_elem * n_files), Y(i,0).n_elem);
      for(int j = std::ceil(burnin_prop * sigma_i.n_elem * n_files); j < sigma_i.n_elem * n_files; j++){
        for(int l = 0; l < Y(i,0).n_elem; l++){
          for(int k = 0; k < Z_samp.n_cols; k++){
            curve_fit(i,0)(j - std::floor(burnin_prop * sigma_i.n_elem * n_files), l) = curve_fit(i,0)(j - std::floor(burnin_prop * sigma_i.n_elem * n_files), l) + (Z_samp(i,k,j) * arma::dot(nu_samp.slice(j).row(k).t() +
              (eta_samp(j,0).slice(k) * X1.row(i).t()), B_obs(i,0).row(l)));
            for(int m = 0; m < chi_samp.n_cols; m++){
              curve_fit(i,0)(j - std::floor(burnin_prop * sigma_i.n_elem * n_files),l) = curve_fit(i,0)(j - std::floor(burnin_prop * sigma_i.n_elem * n_files), l) + (Z_samp(i,k,j) * chi_samp(i,m,j) * arma::dot(phi_samp(j,0).slice(m).row(k).t() +
                (xi_samp(j,k).slice(m) * X1.row(i).t()), B_obs(i,0).row(l)));
            }
          }
        }
      }
    }

    // Get mean curve fit
    arma::field<arma::rowvec> mean_curve_fit(Z_i.n_rows, 1);
    for(int i = 0; i < Z_i.n_rows; i++){
      mean_curve_fit(i,0) = arma::mean(curve_fit(i,0), 0);
    }

    // Get posterior mean of sigma^2
    double mean_sigma = arma::mean(sigma_samp);

    // Calculate Log likelihood
    double log_lik = 0;
    for(int i = 0; i < Z_samp.n_rows; i++){
      for(int j = 0; j < Y(i,0).n_elem; j++){
        log_lik = log_lik + R::dnorm(Y(i,0)(j), mean_curve_fit(i,0)(j),
                                     std::sqrt(mean_sigma), true);
      }
    }

    double tilde_N = 0;
    for(int i = 0; i < Z_samp.n_rows; i++){
      tilde_N = tilde_N + Y(i,0).n_elem;
    }

    // Calculate BIC
    BIC = (2 * log_lik) - (std::log(tilde_N) *((Z_samp.n_rows + phi_samp(0,0).n_cols) * Z_samp.n_cols +
      2 * phi_samp(0,0).n_cols * phi_samp(0,0).n_slices * phi_samp(0,0).n_rows +
      2 + 4 * Z_samp.n_cols + chi_samp.n_rows * chi_samp.n_cols + (phi_samp(0,0).n_slices * Z_samp.n_cols) +
      eta_samp(0,0).n_rows * eta_samp(0,0).n_cols * eta_samp(0,0).n_slices +
      eta_samp(0,0).n_cols * eta_samp(0,0).n_slices));
    if(cov_adj == true){
      BIC = BIC - (std::log(tilde_N) * (2 * eta_samp(0,0).n_rows * eta_samp(0,0).n_cols * eta_samp(0,0).n_slices * phi_samp(0,0).n_slices +
        eta_samp(0,0).n_cols * eta_samp(0,0).n_slices * phi_samp(0,0).n_slices + 2 * eta_samp(0,0).n_cols * eta_samp(0,0).n_slices));
    }
  }

  return(BIC);
}

//' Calculates the Log-Likelihood of a Functional Model
//'
//' Calculates the log-likelihood of the parameters for each iteration for functional models.
//' This function can handle covariate adjusted models as well as non-adjusted models.
//'
//' @name FLLik
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param basis_degree Int containing the degree of B-splines used
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param time Field of vectors containing time points at which the function was observed
//' @param Y Field of vectors containing observed values of the function
//' @param X Matrix of covariates (each row contains the covariates for a single observation) (optional arugment)
//' @param cov_adj Boolean containing whether the model fit had a covariance structure that is covariate-dependent (optional argument)
//' @returns LLik Vector containing the log-likelihood evaluated at each iteration
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{basis_degree}}{must be an integer larger than or equal to 1}
//'   \item{\code{internal_knots}}{must lie in the range of \code{boundary_knots}}
//'   \item{\code{X}}{must have the same number of columns as covariates in the model (D)}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Set Hyperparameters
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//'
//' ## Get CI for mean function
//' LL <- FLLik(dir, n_files, basis_degree, boundary_knots, internal_knots,
//'             time, Y)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Set Hyperparameters
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' LL <- FLLik(dir, n_files, basis_degree, boundary_knots, internal_knots,
//'             time, Y, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Set Hyperparameters
//' Y <- readRDS(system.file("test-data", "Sim_data.RDS", package = "BayesFMMM"))
//' time <- readRDS(system.file("test-data", "time.RDS", package = "BayesFMMM"))
//' dir <- system.file("test-data", "Functional_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' basis_degree <- 3
//' boundary_knots <- c(0, 1000)
//' internal_knots <- c(250, 500, 750)
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' LL <- FLLik(dir, n_files, basis_degree, boundary_knots, internal_knots,
//'             time, Y, X = X, cov_adj = T)
//' @export
// [[Rcpp::export]]
arma::vec FLLik(const std::string dir,
                const int n_files,
                const int basis_degree,
                const arma::vec boundary_knots,
                const arma::vec internal_knots,
                const arma::field<arma::vec> time,
                const arma::field<arma::vec> Y,
                Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                const bool cov_adj = false){

  if(basis_degree <  1){
    Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
  }
  for(int i = 0; i < internal_knots.n_elem; i++){
    if(boundary_knots(0) >= internal_knots(i)){
      Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
    }
    if(boundary_knots(1) <= internal_knots(i)){
      Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
    }
  }
  if(n_files <= 0){
    Rcpp::stop("'n_files' must be greater than 0");
  }

  // Get Nu parameters
  arma::cube nu_i;
  nu_i.load(dir + "Nu0.txt");
  int n_MCMC = nu_i.n_slices;
  arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
  nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
  for(int i = 1; i < n_files; i++){
    nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
    nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                    (nu_i.n_slices)*(i+1) - 1) = nu_i;
  }

  // Get Phi parameters
  arma::field<arma::cube> phi_i;
  phi_i.load(dir + "Phi0.txt");
  arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
  for(int i = 0; i < n_MCMC; i++){
    phi_samp(i,0) = phi_i(i,0);
  }

  for(int i = 1; i < n_files; i++){
    phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
    for(int j = 0; j < n_MCMC; j++){
      phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
    }
  }

  // Get Z parameters
  arma::cube Z_i;
  Z_i.load(dir + "Z0.txt");
  arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
  Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
  for(int i = 1; i < n_files; i++){
    Z_i.load(dir + "Z" + std::to_string(i) +".txt");
    Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
  }

  // Get sigma parameters
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
  sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
  for(int i = 1; i < n_files; i++){
    sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
    sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
  }

  // Get chi parameters
  arma::cube chi_i;
  chi_i.load(dir + "Chi0.txt");
  arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
  chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
  for(int i = 1; i < n_files; i++){
    chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
    chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                     (chi_i.n_slices)*(i+1) - 1) = chi_i;
  }

  // Get eta parameters
  arma::field<arma::cube> eta_samp(n_MCMC * n_files, 1);
  if(X.isNull()){
    for(int i = 0; i < n_MCMC * n_files; i ++){
      eta_samp(i,0) = arma::zeros(nu_samp.n_cols, 1, nu_samp.n_rows);
    }
  }else{
    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    for(int i = 0; i < n_MCMC; i++){
      eta_samp(i,0) = eta_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        eta_samp((i * n_MCMC) + j, 0) = eta_i(j,0);
      }
    }
  }

  arma::field<arma::cube> xi_samp(n_MCMC * n_files, nu_samp.n_rows);
  if(cov_adj == false){
    for(int k = 0; k < nu_samp.n_rows; k++){
      for(int i = 0; i < n_MCMC * n_files; i ++){
        xi_samp(i,k) = arma::zeros(nu_samp.n_cols, eta_samp(0,0).n_cols, phi_samp(0,0).n_slices);
      }
    }
  }else{
    arma::field<arma::cube> xi_i;
    xi_i.load(dir + "Xi0.txt");
    for(int k = 0; k < nu_samp.n_rows; k++){
      for(int i = 0; i < n_MCMC; i++){
        xi_samp(i,k) = xi_i(i,k);
      }
    }

    for(int i = 1; i < n_files; i++){
      xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int j = 0; j < n_MCMC; j++){
          xi_samp((i * n_MCMC) + j, k) = xi_i(j,k);
        }
      }
    }
  }

  // initialize X
  arma::mat X1;
  if(X.isNotNull()) {
    Rcpp::NumericMatrix X_(X);
    X1 = Rcpp::as<arma::mat>(X_);
    if(X1.n_cols != eta_samp(0,0).n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }

  }else{
    X1 = arma::zeros(Y.n_rows, 1);
  }


  // Make spline basis
  arma::field<arma::mat> B_obs(Z_samp.n_rows, 1);
  splines2::BSpline bspline2;
  for(int i = 0; i < Z_samp.n_rows; i++){
    bspline2 = splines2::BSpline(time(i,0),  internal_knots, basis_degree,
                                 boundary_knots);
    // Get Basis matrix (time2 x Phi.n_cols)
    arma::mat bspline_mat2{bspline2.basis(true)};
    // Make B_obs
    B_obs(i,0) = bspline_mat2;
  }
  arma::vec LLik = arma::zeros(nu_samp.n_slices);
  for(int i = 0; i < nu_samp.n_slices; i++){
    LLik(i) =  BayesFMMM::calcLikelihoodCovariateAdj(Y, B_obs, nu_samp.slice(i),
         eta_samp(i,0), phi_samp(i,0), xi_samp, Z_samp.slice(i),
         chi_samp.slice(i), i, X1, sigma_samp(i));
  }

 return(LLik);
}

//' Calculates the AIC of a multivariate model
//'
//' @name MV_Model_AIC
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param n_MCMC Int containing the number of saved MCMC iterations per file
//' @param Y Matrix of observed vectors (each row is an observation)
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @returns AIC Double containing AIC value
//' @export
// [[Rcpp::export]]
double MV_Model_AIC(const std::string dir,
                    const int n_files,
                    const int n_MCMC,
                    const arma::mat Y,
                    const double burnin_prop = 0.2){

  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }
  if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }

  // Get Nu parameters
  arma::cube nu_i;
  nu_i.load(dir + "Nu0.txt");
  arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
  nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
  for(int i = 1; i < n_files; i++){
    nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
    nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                    (nu_i.n_slices)*(i+1) - 1) = nu_i;
  }

  // Get Phi parameters
  arma::field<arma::cube> phi_i;
  phi_i.load(dir + "Phi0.txt");
  arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
  for(int i = 0; i < n_MCMC; i++){
    phi_samp(i,0) = phi_i(i,0);
  }

  for(int i = 1; i < n_files; i++){
    phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
    for(int j = 0; j < n_MCMC; j++){
      phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
    }
  }

  // Get Z parameters
  arma::cube Z_i;
  Z_i.load(dir + "Z0.txt");
  arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
  Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
  for(int i = 1; i < n_files; i++){
    Z_i.load(dir + "Z" + std::to_string(i) +".txt");
    Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
  }

  // Get sigma parameters
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
  sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
  for(int i = 1; i < n_files; i++){
    sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
    sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
  }

  // Get chi parameters
  arma::cube chi_i;
  chi_i.load(dir + "Chi0.txt");
  arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
  chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
  for(int i = 1; i < n_files; i++){
    chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
    chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                     (chi_i.n_slices)*(i+1) - 1) = chi_i;
  }


  // Get posterior mean of sigma^2
  double mean_sigma = arma::mean(sigma_samp);

  // Get individual mean vector
  arma::field<arma::mat> fit(Z_i.n_rows, 1);
  arma::rowvec Z_ph(Z_i.n_cols, arma::fill::zeros);
  for(int i = 0; i < Z_i.n_rows; i++){
    fit(i,0) = arma::zeros(std::ceil((1 - burnin_prop) * sigma_i.n_elem * n_files), Y.n_cols);
    for(int j = std::floor(burnin_prop * sigma_i.n_elem * n_files); j < sigma_i.n_elem * n_files; j++){
      Z_ph = Z_samp(arma::span(i), arma::span::all, arma::span(j));
      fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) = Z_ph * nu_samp.slice(j);
      for(int k = 0; k < chi_samp.n_cols; k++){
        fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) = fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) + (chi_samp(i, k, j) *
          Z_ph * phi_samp(j,0).slice(k));
      }
    }
  }

  // Get mean curve fit
  arma::mat mean_fit = arma::zeros(Y.n_rows, Y.n_cols);
  for(int i = 0; i < Z_i.n_rows; i++){
    mean_fit.row(i) = arma::mean(fit(i,0), 0);
  }

  // Calculate Log likelihood
  double log_lik = 0;
  for(int i = 0; i < Z_samp.n_rows; i++){
    for(int j = 0; j < Y.n_cols; j++){
      log_lik = log_lik + R::dnorm(Y(i,j), mean_fit(i,j),
                                   std::sqrt(mean_sigma), true);
    }
  }


  // Calculate AIC
  double AIC =  2*((Z_samp.n_rows + phi_samp(0,0).n_cols) * Z_samp.n_cols +
                   2 * phi_samp(0,0).n_cols * phi_samp(0,0).n_slices * phi_samp(0,0).n_rows +
                   2 + 4 * Z_samp.n_cols + chi_samp.n_rows * chi_samp.n_cols + (phi_samp(0,0).n_slices*Z_samp.n_cols)) - (2 * log_lik);
  return(AIC);
}

//' Calculates the BIC of a multivariate model
//'
//' @name MV_Model_BIC
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param n_MCMC Int containing the number of saved MCMC iterations per file
//' @param Y Matrix of observed vectors (each row is an observation)
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @returns BIC Double containing BIC value
//' @export
// [[Rcpp::export]]
double MV_Model_BIC(const std::string dir,
                    const int n_files,
                    const int n_MCMC,
                    const arma::mat Y,
                    const double burnin_prop = 0.2){


  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }
  if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }

  // Get Nu parameters
  arma::cube nu_i;
  nu_i.load(dir + "Nu0.txt");
  arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
  nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
  for(int i = 1; i < n_files; i++){
    nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
    nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                    (nu_i.n_slices)*(i+1) - 1) = nu_i;
  }

  // Get Phi parameters
  arma::field<arma::cube> phi_i;
  phi_i.load(dir + "Phi0.txt");
  arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
  for(int i = 0; i < n_MCMC; i++){
    phi_samp(i,0) = phi_i(i,0);
  }

  for(int i = 1; i < n_files; i++){
    phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
    for(int j = 0; j < n_MCMC; j++){
      phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
    }
  }

  // Get Z parameters
  arma::cube Z_i;
  Z_i.load(dir + "Z0.txt");
  arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
  Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
  for(int i = 1; i < n_files; i++){
    Z_i.load(dir + "Z" + std::to_string(i) +".txt");
    Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
  }

  // Get sigma parameters
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
  sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
  for(int i = 1; i < n_files; i++){
    sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
    sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
  }

  // Get chi parameters
  arma::cube chi_i;
  chi_i.load(dir + "Chi0.txt");
  arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
  chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
  for(int i = 1; i < n_files; i++){
    chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
    chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                     (chi_i.n_slices)*(i+1) - 1) = chi_i;
  }

  // Get posterior mean of sigma^2
  double mean_sigma = arma::mean(sigma_samp);

  // Get individual mean vector
  arma::field<arma::mat> fit(Z_i.n_rows, 1);
  arma::rowvec Z_ph(Z_i.n_cols, arma::fill::zeros);
  for(int i = 0; i < Z_i.n_rows; i++){
    fit(i,0) = arma::zeros(std::ceil((1 - burnin_prop) * sigma_i.n_elem * n_files), Y.n_cols);
    for(int j = std::floor(burnin_prop * sigma_i.n_elem * n_files); j < sigma_i.n_elem * n_files; j++){
      Z_ph = Z_samp(arma::span(i), arma::span::all, arma::span(j));
      fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) = Z_ph * nu_samp.slice(j);
      for(int k = 0; k < chi_samp.n_cols; k++){
        fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) = fit(i,0).row(j - std::floor(burnin_prop * sigma_i.n_elem * n_files)) + (chi_samp(i, k, j) *
          Z_ph * phi_samp(j,0).slice(k));
      }
    }
  }

  // Get mean curve fit
  arma::mat mean_fit = arma::zeros(Y.n_rows, Y.n_cols);
  for(int i = 0; i < Z_i.n_rows; i++){
    mean_fit.row(i) = arma::mean(fit(i,0), 0);
  }

  // Calculate Log likelihood
  double log_lik = 0;
  for(int i = 0; i < Z_samp.n_rows; i++){
    for(int j = 0; j < Y.n_cols; j++){
      log_lik = log_lik + R::dnorm(Y(i,j), mean_fit(i,j),
                                   std::sqrt(mean_sigma), true);
    }
  }

  // Calculate BIC
  double BIC = (2 * log_lik) - (std::log(Y.n_rows) *
                ((Z_samp.n_rows + phi_samp(0,0).n_cols) * Z_samp.n_cols +
                2 * phi_samp(0,0).n_cols * phi_samp(0,0).n_slices * phi_samp(0,0).n_rows +
                2 + 4 * Z_samp.n_cols + chi_samp.n_rows * chi_samp.n_cols + (phi_samp(0,0).n_slices*Z_samp.n_cols)));
  return(BIC);
}

//' Calculates the DIC of a functional model
//'
//' @name Model_DIC
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param n_MCMC Int containing the number of saved MCMC iterations per file
//' @param Y Matrix of observed vectors (each row is an observation)
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @returns DIC Double containing DIC value
//' @export
// [[Rcpp::export]]
double MV_Model_DIC(const std::string dir,
                    const int n_files,
                    const int n_MCMC,
                    const arma::mat Y,
                    const double burnin_prop = 0.2){
  if(burnin_prop < 0){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }
  if(burnin_prop >= 1){
    Rcpp::stop("'burnin_prop' must be between 0 and 1");
  }

  // Get Nu parameters
  arma::cube nu_i;
  nu_i.load(dir + "Nu0.txt");
  arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
  nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
  for(int i = 1; i < n_files; i++){
    nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
    nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                    (nu_i.n_slices)*(i+1) - 1) = nu_i;
  }

  // Get Phi parameters
  arma::field<arma::cube> phi_i;
  phi_i.load(dir + "Phi0.txt");
  arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
  for(int i = 0; i < n_MCMC; i++){
    phi_samp(i,0) = phi_i(i,0);
  }

  for(int i = 1; i < n_files; i++){
    phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
    for(int j = 0; j < n_MCMC; j++){
      phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
    }
  }

  // Get Z parameters
  arma::cube Z_i;
  Z_i.load(dir + "Z0.txt");
  arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
  Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
  for(int i = 1; i < n_files; i++){
    Z_i.load(dir + "Z" + std::to_string(i) +".txt");
    Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
  }

  // Get sigma parameters
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
  sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
  for(int i = 1; i < n_files; i++){
    sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
    sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
  }

  // Get chi parameters
  arma::cube chi_i;
  chi_i.load(dir + "Chi0.txt");
  arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
  chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
  for(int i = 1; i < n_files; i++){
    chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
    chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                     (chi_i.n_slices)*(i+1) - 1) = chi_i;
  }


  double expected_log_f = 0;
  for(int i = std::floor(burnin_prop *nu_samp.n_slices) ; i < nu_samp.n_slices; i++){
    expected_log_f = expected_log_f + BayesFMMM::calcLikelihoodMV(Y, nu_samp.slice(i),
                                                                  phi_samp(i,0), Z_samp.slice(i),
                                                                  chi_samp.slice(i), sigma_samp(i));
  }
  expected_log_f = expected_log_f / std::ceil((1-burnin_prop) *nu_samp.n_slices);

  double f_hat = 0;
  double f_hat_i = 0;
  for(int i = 0; i < Z_samp.n_rows; i++){
    f_hat_i = 0;
    for(int n = std::floor(burnin_prop *nu_samp.n_slices); n < nu_samp.n_slices; n++){
      f_hat_i = f_hat_i + BayesFMMM::calcDIC2MV(Y.row(i), nu_samp.slice(n), phi_samp(n,0),
                                                Z_samp.slice(n), chi_samp.slice(n), i,
                                                sigma_samp(n));
    }
    f_hat = f_hat + std::log(f_hat_i / std::ceil((1-burnin_prop) *nu_samp.n_slices));
  }

  double DIC = (2 * f_hat) - (4 * expected_log_f);
  return(DIC);
}

//' Calculates the Log-Likelihood of a Multivariate Model
//'
//' Calculates the log-likelihood of the parameters for each iteration of a multivariate model.
//' This function can handle covariate adjusted models as well as non-adjusted models.
//'
//'
//' @name MVLLik
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param Y Matrix of observed vectors (each row is an observation)
//' @param X Matrix of covariates (each row contains the covariates for a single observation) (optional arugment)
//' @param cov_adj Boolean containing whether the model fit had a covariance structure that is covariate-dependent (optional argument)
//' @returns LLik Vector containing the log-likelihood evaluated at each iteration
//'
//' @section Warning:
//' The following must be true:
//' \describe{
//'   \item{\code{n_files}}{must be an integer larger than or equal to 1}
//'   \item{\code{X}}{must have the same number of columns as covariates in the model (D)}
//' }
//'
//' @examples
//' #########################
//' ### Not Covariate Adj ###
//' #########################
//'
//' ## Set Hyperparameters
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//'
//' ## Get CI for mean function
//' LL <- MVLLik(dir, n_files, Y)
//'
//' #####################
//' ### Covariate Adj ###
//' #####################
//'
//' ## Set Hyperparameters
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' LL <- MVLLik(dir, n_files, Y, X = X)
//'
//' #####################################################################
//' ### Covariate Adj  (with Covariate-depenent covariance structure) ###
//' #####################################################################
//'
//' ## Set Hyperparameters
//' Y <- readRDS(system.file("test-data", "MVSim_data.RDS", package = "BayesFMMM"))
//' dir <- system.file("test-data", "Multivariate_trace", "", package = "BayesFMMM")
//' n_files <- 1
//' X <- matrix(seq(-2, 2, 0.2), ncol = 1)
//'
//' ## Get CI for mean function
//' LL <- MVLLik(dir, n_files, Y, X = X, cov_adj = T)
//' @export
// [[Rcpp::export]]
arma::vec MVLLik(const std::string dir,
                 const int n_files,
                 const arma::mat Y,
                 Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                 const bool cov_adj = false){

  if(n_files <= 0){
    Rcpp::stop("'n_files' must be greater than 0");
  }

  // Get Nu parameters
  arma::cube nu_i;
  nu_i.load(dir + "Nu0.txt");
  int n_MCMC = nu_i.n_slices;
  arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
  nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
  for(int i = 1; i < n_files; i++){
    nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
    nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                    (nu_i.n_slices)*(i+1) - 1) = nu_i;
  }

  // Get Phi parameters
  arma::field<arma::cube> phi_i;
  phi_i.load(dir + "Phi0.txt");
  arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
  for(int i = 0; i < n_MCMC; i++){
    phi_samp(i,0) = phi_i(i,0);
  }

  for(int i = 1; i < n_files; i++){
    phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
    for(int j = 0; j < n_MCMC; j++){
      phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
    }
  }

  // Get Z parameters
  arma::cube Z_i;
  Z_i.load(dir + "Z0.txt");
  arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
  Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
  for(int i = 1; i < n_files; i++){
    Z_i.load(dir + "Z" + std::to_string(i) +".txt");
    Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
  }

  // Get sigma parameters
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
  sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
  for(int i = 1; i < n_files; i++){
    sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
    sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
  }

  // Get chi parameters
  arma::cube chi_i;
  chi_i.load(dir + "Chi0.txt");
  arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
  chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
  for(int i = 1; i < n_files; i++){
    chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
    chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                     (chi_i.n_slices)*(i+1) - 1) = chi_i;
  }

  // Get eta parameters
  arma::field<arma::cube> eta_samp(n_MCMC * n_files, 1);
  if(X.isNull()){
    for(int i = 0; i < n_MCMC * n_files; i ++){
      eta_samp(i,0) = arma::zeros(nu_samp.n_cols, 1, nu_samp.n_rows);
    }
  }else{
    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    for(int i = 0; i < n_MCMC; i++){
      eta_samp(i,0) = eta_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        eta_samp((i * n_MCMC) + j, 0) = eta_i(j,0);
      }
    }
  }

  arma::field<arma::cube> xi_samp(n_MCMC * n_files, nu_samp.n_rows);
  if(cov_adj == false){
    for(int k = 0; k < nu_samp.n_rows; k++){
      for(int i = 0; i < n_MCMC * n_files; i++){
        xi_samp(i,k) = arma::zeros(nu_samp.n_cols, 1, phi_samp(0,0).n_slices);
      }
    }
  }else{
    arma::field<arma::cube> xi_i;
    xi_i.load(dir + "Xi0.txt");
    for(int k = 0; k < nu_samp.n_rows; k++){
      for(int i = 0; i < n_MCMC; i++){
        xi_samp(i,k) = xi_i(i,k);
      }
    }

    for(int i = 1; i < n_files; i++){
      xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int j = 0; j < n_MCMC; j++){
          xi_samp((i * n_MCMC) + j, k) = xi_i(j,k);
        }
      }
    }
  }

  // initialize X
  arma::mat X1;
  if(X.isNotNull()) {
    Rcpp::NumericMatrix X_(X);
    X1 = Rcpp::as<arma::mat>(X_);
    if(X1.n_cols != eta_samp(0,0).n_cols){
      Rcpp::stop("The number of columns in 'X' must be equal to the number of covariates in the model");
    }
  }else{
    X1 = arma::zeros(Y.n_rows, 1);
  }

  arma::vec LLik = arma::zeros(nu_samp.n_slices);
  for(int i = 0; i < nu_samp.n_slices; i++){
    LLik(i) =  BayesFMMM::calcLikelihoodMVCovariateAdj(Y, nu_samp.slice(i), eta_samp(i,0),
         phi_samp(i,0), xi_samp, Z_samp.slice(i),
         chi_samp.slice(i), i, X1, sigma_samp(i));
  }

  return(LLik);
}


//' Conditional Predictive Ordinates
//'
//' Calculates the Conditional Predictive Ordinates for functional models.
//' This function can handle covariate adjusted models as well as non-adjusted models.
//'
//' @name ConditionalPredictiveOrdinates
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param n_MCMC Int containing the number of saved MCMC iterations per file
//' @param basis_degree Int containing the degree of B-splines used
//' @param boundary_knots Vector containing the boundary points of our index domain of interest
//' @param internal_knots Vector location of internal knots for B-splines
//' @param time Field of vectors containing time points at which the function was observed
//' @param Y Field of vectors containing observed values of the function
//' @param burnin_prop Double containing proportion of MCMC samples to discard
//' @param X Matrix of covariates (each row contains the covariates for a single observation) (optional arugment)
//' @param cov_adj Boolean containing whether the model fit had a covariance structure that is covariate-dependent (optional argument)
//' @param log_CPO Boolean conatining whether or not CPO is returned on the log scale (optional argument)
//' @returns CPO Vector containing the CPO for each observation
//' @export
// [[Rcpp::export]]
arma::vec ConditionalPredictiveOrdinates(const std::string dir,
                                         const int n_files,
                                         const int n_MCMC,
                                         const int basis_degree,
                                         const arma::vec boundary_knots,
                                         const arma::vec internal_knots,
                                         const arma::field<arma::vec> time,
                                         const arma::field<arma::vec> Y,
                                         const double burnin_prop = 0.2,
                                         Rcpp::Nullable<Rcpp::NumericMatrix> X = R_NilValue,
                                         const bool cov_adj = false,
                                         const bool log_CPO = true){

  bool mean_adj = false;
  if(X.isNotNull()){
    mean_adj = true;
  }
  if(basis_degree <  1){
    Rcpp::stop("'basis_degree' must be an integer greater than or equal to 1");
  }
  for(int i = 0; i < internal_knots.n_elem; i++){
    if(boundary_knots(0) >= internal_knots(i)){
      Rcpp::stop("at least one element in 'internal_knots' is less than or equal to first boundary knot");
    }
    if(boundary_knots(1) <= internal_knots(i)){
      Rcpp::stop("at least one element in 'internal_knots' is more than or equal to second boundary knot");
    }
  }

  // Get Nu parameters
  arma::cube nu_i;
  nu_i.load(dir + "Nu0.txt");
  arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
  nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
  for(int i = 1; i < n_files; i++){
    nu_i.load(dir + "Nu" + std::to_string(i) +".txt");
    nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1,
                    (nu_i.n_slices)*(i+1) - 1) = nu_i;
  }
  // Get Phi parameters
  arma::field<arma::cube> phi_i;
  phi_i.load(dir + "Phi0.txt");
  arma::field<arma::cube> phi_samp(n_MCMC * n_files, 1);
  for(int i = 0; i < n_MCMC; i++){
    phi_samp(i,0) = phi_i(i,0);
  }

  for(int i = 1; i < n_files; i++){
    phi_i.load(dir + "Phi" + std::to_string(i) +".txt");
    for(int j = 0; j < n_MCMC; j++){
      phi_samp((i * n_MCMC) + j, 0) = phi_i(j,0);
    }
  }

  // Get Z parameters
  arma::cube Z_i;
  Z_i.load(dir + "Z0.txt");
  arma::cube Z_samp = arma::zeros(Z_i.n_rows, Z_i.n_cols, Z_i.n_slices * n_files);
  Z_samp.subcube(0, 0, 0, Z_i.n_rows-1, Z_i.n_cols-1, Z_i.n_slices-1) = Z_i;
  for(int i = 1; i < n_files; i++){
    Z_i.load(dir + "Z" + std::to_string(i) +".txt");
    Z_samp.subcube(0, 0,  Z_i.n_slices*i, Z_i.n_rows-1, Z_i.n_cols-1, (Z_i.n_slices)*(i+1) - 1) = Z_i;
  }

  // Get sigma parameters
  arma::vec sigma_i;
  sigma_i.load(dir + "Sigma0.txt");
  arma::vec sigma_samp = arma::zeros(sigma_i.n_elem * n_files);
  sigma_samp.subvec(0, sigma_i.n_elem - 1) = sigma_i;
  for(int i = 1; i < n_files; i++){
    sigma_i.load(dir + "Sigma" + std::to_string(i) +".txt");
    sigma_samp.subvec(sigma_i.n_elem *i, (sigma_i.n_elem *(i + 1)) - 1) = sigma_i;
  }

  // Get chi parameters
  arma::cube chi_i;
  chi_i.load(dir + "Chi0.txt");
  arma::cube chi_samp = arma::zeros(chi_i.n_rows, chi_i.n_cols, chi_i.n_slices * n_files);
  chi_samp.subcube(0, 0, 0, chi_i.n_rows-1, chi_i.n_cols-1, chi_i.n_slices-1) = chi_i;
  for(int i = 1; i < n_files; i++){
    chi_i.load(dir + "Chi" + std::to_string(i) +".txt");
    chi_samp.subcube(0, 0,  chi_i.n_slices*i, chi_i.n_rows-1, chi_i.n_cols-1,
                     (chi_i.n_slices)*(i+1) - 1) = chi_i;
  }

  // Get eta parameters
  arma::field<arma::cube> eta_samp(n_MCMC * n_files, 1);
  if(mean_adj == false){
    for(int i = 0; i < n_MCMC * n_files; i ++){
      eta_samp(i,0) = arma::zeros(nu_samp.n_cols, 1, nu_samp.n_rows);
    }
  }else{
    arma::field<arma::cube> eta_i;
    eta_i.load(dir + "Eta0.txt");
    for(int i = 0; i < n_MCMC; i++){
      eta_samp(i,0) = eta_i(i,0);
    }

    for(int i = 1; i < n_files; i++){
      eta_i.load(dir + "Eta" + std::to_string(i) +".txt");
      for(int j = 0; j < n_MCMC; j++){
        eta_samp((i * n_MCMC) + j, 0) = eta_i(j,0);
      }
    }
  }

  // initialize X
  arma::mat X1;
  if(X.isNotNull()) {
    Rcpp::NumericMatrix X_(X);
    X1 = Rcpp::as<arma::mat>(X_);
  }else{
    X1 = arma::zeros(Y.n_rows, 1);
  }

  arma::field<arma::cube> xi_samp(n_MCMC * n_files, nu_samp.n_rows);
  if(cov_adj == false){
    for(int k = 0; k < nu_samp.n_rows; k++){
      for(int i = 0; i < n_MCMC * n_files; i ++){
        xi_samp(i,k) = arma::zeros(nu_samp.n_cols, X1.n_cols, phi_samp(0,0).n_slices);
      }
    }
  }else{
    arma::field<arma::cube> xi_i;
    xi_i.load(dir + "Xi0.txt");
    for(int k = 0; k < nu_samp.n_rows; k++){
      for(int i = 0; i < n_MCMC; i++){
        xi_samp(i,k) = xi_i(i,k);
      }
    }
    for(int i = 1; i < n_files; i++){
      xi_i.load(dir + "Xi" + std::to_string(i) +".txt");
      for(int k = 0; k < nu_samp.n_rows; k++){
        for(int j = 0; j < n_MCMC; j++){
          xi_samp((i * n_MCMC) + j, k) = xi_i(j,k);
        }
      }
    }
  }


  // Make spline basis
  arma::field<arma::mat> B_obs(Z_samp.n_rows, 1);
  splines2::BSpline bspline2;
  for(int i = 0; i < Z_samp.n_rows; i++){
    bspline2 = splines2::BSpline(time(i,0),  internal_knots, basis_degree,
                                 boundary_knots);
    // Get Basis matrix (time2 x Phi.n_cols)
    arma::mat bspline_mat2{bspline2.basis(true)};
    // Make B_obs
    B_obs(i,0) = bspline_mat2;
  }

  arma::vec CPO =  BayesFMMM::calcLikelihoodCPO(Y, B_obs, nu_samp,
       eta_samp, phi_samp, xi_samp, Z_samp,
       chi_samp, X1, sigma_samp, n_MCMC * n_files, burnin_prop);
  if(log_CPO == false){
    CPO = arma::exp(CPO);
  }

  return(CPO);
}
