#include <RcppArmadillo.h>
#include <cmath>
#include <testthat.h>
#include "BSplines.H"

//' Tests creation of tensor product B-splines for multivariate functional data
//'
//' @name TestBSpline
arma::mat TestBSplineTensor(){
  arma::field<arma::mat> t_obs1(2,1);
  t_obs1(0,0) = arma::zeros(100,2);
  t_obs1(0,0).col(0) =  arma::regspace(0, 10, 990);
  t_obs1(0,0).col(1) =  arma::regspace(0, 10, 990);
  t_obs1(1,0) =  t_obs1(0,0);

  splines2::BSpline bspline;
  arma::field<arma::vec> internal_knots(2,1);
  internal_knots(0,0) = {250, 500, 750};
  internal_knots(1,0) = {250, 500, 750};
  arma::mat boundary_knots = {{0,990}, {0,990}};

  arma::vec basis_degree = {3,3};

  arma::field<arma::mat> B = TensorBSpline(t_obs1, 2, basis_degree,
                                           boundary_knots, internal_knots);

  return B(0,0);
}


//' Tests creation of P matrix used for updating Nu
//'
//' @name TestBSpline
arma::mat TestPMat(){
  arma::field<arma::mat> t_obs1(2,1);
  t_obs1(0,0) = arma::zeros(100,2);
  t_obs1(0,0).col(0) =  arma::regspace(0, 10, 990);
  t_obs1(0,0).col(1) =  arma::regspace(0, 10, 990);
  t_obs1(1,0) =  t_obs1(0,0);

  splines2::BSpline bspline;
  arma::field<arma::vec> internal_knots(2,1);
  internal_knots(0,0) = {250, 500, 750};
  internal_knots(1,0) = {250, 500, 750};
  arma::mat boundary_knots = {{0,990}, {0,990}};

  arma::vec basis_degree = {3,3};

  arma::mat P = GetP(basis_degree,internal_knots);

  return P;
}


// Tests creation of multivariate B-splines
context("Tensor B-Spline unit tests") {

  test_that("creation of bivariate spline") {
    Rcpp::Environment base("package:base");
    Rcpp::Function sys_file = base["system.file"];
    Rcpp::StringVector path = sys_file("inst", "test-data", "Tensor_BSpline.txt",
                                       Rcpp::_["package"] = "BayesFPMM");
    std::string string_path = Rcpp::as<std::string>(path[0]);
    arma::mat B_true;
    B_true.load(string_path);
    expect_true(arma::approx_equal(TestBSplineTensor(), B_true, "absdiff", 1e-7));
  }
}

// Tests creation of multivariate B-splines
context("Tensor P matrix for Nu") {

  test_that("creation of P matrix") {
    Rcpp::Environment base("package:base");
    Rcpp::Function sys_file = base["system.file"];
    Rcpp::StringVector path = sys_file("inst", "test-data", "P_mat.txt",
                                       Rcpp::_["package"] = "BayesFPMM");
    std::string string_path = Rcpp::as<std::string>(path[0]);
    arma::mat P_true;
    P_true.load(string_path);
    expect_true(arma::approx_equal(TestPMat(), P_true, "absdiff", 1e-7));
  }
}
