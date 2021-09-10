#include <RcppArmadillo.h>
#include <splines2Armadillo.h>
#include <cmath>

//' Calculates the Pointwise credible interval for the mean
//'
//' @name GetMeanCI_Pw
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param time Vector containing time points of interest
//' @param k Int containing the cluster group of which you want to get the credible interval for
//' @return CI list containing the 97.5th , 50th, and 2.5th pointwise functions
//' @export
// [[Rcpp::export]]
Rcpp::List GetMeanCI_PW(const std::string dir,
                        const int n_files,
                        const arma::vec time,
                        const int k){
  arma::cube nu_i;
  nu_i.load(dir + "Nu0.txt");
  arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
  nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
  for(int i = 1; i < n_files; i++){
    nu_i.load(dir + "nu" + std::to_string(i) +".txt");
    nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1, (nu_i.n_slices)*(i+1) - 1) = nu_i;
  }
  splines2::BSpline bspline;
  bspline = splines2::BSpline(time, nu_i.n_cols);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat {bspline.basis(true)};
  // Make B_obs
  arma::mat B = bspline_mat;

  arma::mat f_samp = arma::zeros(nu_samp.n_slices, time.n_elem);
  for(int i = 0; i < nu_samp.n_slices; i++){
    f_samp.row(i) = (B * nu_samp.slice(i).row(k-1).t()).t();
  }

  // Initialize placeholders
  arma::vec CI_975 = arma::zeros(time.n_elem);
  arma::vec CI_50 = arma::zeros(time.n_elem);
  arma::vec CI_025 = arma::zeros(time.n_elem);

  arma::vec p = {0.025, 0.5, 0.975};
  arma::vec q = arma::zeros(3);

  for(int i = 0; i < time.n_elem; i++){
    q = arma::quantile(f_samp.col(i), p);
    CI_025(i) = q(0);
    CI_50(i) = q(1);
    CI_975(i) = q(2);
  }
  Rcpp::List CI =  Rcpp::List::create(Rcpp::Named("CI_975", CI_975),
                                      Rcpp::Named("CI_50", CI_50),
                                      Rcpp::Named("CI_025", CI_025));
  return(CI);
}

//' Calculates the simultaneous credible interval for the mean
//'
//' @name GetMeanCI_S
//' @param dir String containing the directory where the MCMC files are located
//' @param n_files Int containing the number of files per parameter
//' @param time Vector containing time points of interest
//' @param k Int containing the cluster group of which you want to get the credible interval for
//' @return CI list containing the 97.5th , 50th, and 2.5th simultaneous functions
//' @export
// [[Rcpp::export]]
Rcpp::List GetMeanCI_S(const std::string dir,
                       const int n_files,
                       const arma::vec time,
                       const int k){
  arma::cube nu_i;
  nu_i.load(dir + "Nu0.txt");
  arma::cube nu_samp = arma::zeros(nu_i.n_rows, nu_i.n_cols, nu_i.n_slices * n_files);
  nu_samp.subcube(0, 0, 0, nu_i.n_rows-1, nu_i.n_cols-1, nu_i.n_slices-1) = nu_i;
  for(int i = 1; i < n_files; i++){
    nu_i.load(dir + "nu" + std::to_string(i) +".txt");
    nu_samp.subcube(0, 0,  nu_i.n_slices*i, nu_i.n_rows-1, nu_i.n_cols-1, (nu_i.n_slices)*(i+1) - 1) = nu_i;
  }
  splines2::BSpline bspline;
  bspline = splines2::BSpline(time, nu_i.n_cols);
  // Get Basis matrix (100 x 8)
  arma::mat bspline_mat {bspline.basis(true)};
  // Make B_obs
  arma::mat B = bspline_mat;

  arma::mat f_samp = arma::zeros(nu_samp.n_slices, time.n_elem);
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

  // Initialize placeholders
  arma::vec CI_975 = arma::zeros(time.n_elem);
  arma::vec CI_50 = arma::zeros(time.n_elem);
  arma::vec CI_025 = arma::zeros(time.n_elem);

  arma::vec p = {0.95};
  arma::vec q = arma::zeros(1);
  q = arma::quantile(C, p);

  for(int i = 0; i < time.n_elem; i++){

    CI_025(i) = f_mean(i) - q(0) * f_sd(i);
    CI_50(i) = f_mean(i);
    CI_975(i) =  f_mean(i) + q(0) * f_sd(i);
  }
  Rcpp::List CI =  Rcpp::List::create(Rcpp::Named("CI_975", CI_975),
                                      Rcpp::Named("CI_50", CI_50),
                                      Rcpp::Named("CI_025", CI_025));
  return(CI);
}
