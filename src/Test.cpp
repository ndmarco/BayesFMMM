#include <RcppArmadillo.h>
#include <cmath>
#include <splines2Armadillo.h>
#include <BayesFPMM.h>


//
// //' Tests updating Pi
// //'
// //' @name TestUpdatePi
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdatePi()
// {
//   // Make Z matrix
//   arma::mat Z = arma::randi<arma::mat>(1000, 3, arma::distr_param(0,1));
//   double alpha = 1;
//   arma::mat pi = arma::zeros(3, 100);
//
//   for(int i = 0; i < 100; i ++)
//   {
//     updatePi(alpha, Z, i, 100,  pi);
//   }
//   arma::vec prob = arma::zeros(3);
//   for(int i = 0; i < 3; i++)
//   {
//     prob(i) = arma::accu(Z.col(i)) / 1000;
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("pi", pi),
//                                       Rcpp::Named("prob", prob));
//   return mod;
// }
//
//
// //' Tests updating Phi
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdatePhi()
// {
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//
//   for(int i = 0; i < 100; i++)
//   {
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   Z.col(0) = arma::vec(100, arma::fill::ones);
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {0.95, 0.5, 0.5};
//
//   arma::field<arma::cube> Phi_samp(100, 1);
//   for(int i = 0; i < 100 ; i++){
//     Phi_samp(i,0) = arma::zeros(Phi.n_rows, Phi.n_cols, Phi.n_slices);
//   }
//   arma::vec m_1(Phi.n_cols, arma::fill::zeros);
//   arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
//   arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
//   gamma = gamma * 10;
//   arma::vec tilde_tau = {2, 2.5, 3, 5, 10};
//   for(int i = 0; i < 100; i++){
//     updatePhi(y_obs, B_obs, nu, gamma, tilde_tau, Z, chi,
//              sigma_sq, i, 100, m_1, M_1, Phi_samp);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Phi", Phi),
//                                       Rcpp::Named("Phi_samp", Phi_samp));
//   return mod;
// }
//
//
// //' Tests updating Delta
// //'
// //' @name TestUpdateDelta
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateDelta1(){
//   // Specify hyperparameters
//   arma::vec a_12 = {2, 2};
//   // Make Delta vector
//   arma::vec Delta = arma::zeros(5);
//   for(int i=0; i < 5; i++){
//     Delta(i) = R::rgamma(2, 1);
//   }
//   // Make Gamma cube
//   arma::cube Gamma(3,8,5);
//   for(int i=0; i < 5; i++){
//     for(int j = 0; j < 3; j++){
//       for(int k = 0; k < 8; k++){
//         Gamma(j,k,i) =  R::rgamma(1.5, 1/1.5);
//       }
//     }
//   }
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   arma::mat delta = arma::ones(5,10000);
//   for(int m = 0; m < 10000; m++){
//     double tau  = 1;
//     for(int i=0; i < 5; i++){
//       tau = tau * Delta(i);
//       for(int j=0; j < 3; j++){
//         for(int k=0; k < 8; k++){
//           Phi(j,k,i) = R::rnorm(0, (1/ std::pow(Gamma(j,k,i)*tau, 0.5)));
//         }
//       }
//     }
//     BayesFPMM::updateDelta(Phi, Gamma, a_12, m, 10000, delta);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("delta", Delta),
//                                       Rcpp::Named("gamma", Gamma),
//                                       Rcpp::Named("phi", Phi),
//                                       Rcpp::Named("delta_samp", delta));
//   return mod;
// }
// //
// //' Tests updating A
// //'
// //' @name TestUpdateA
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateA1(){
//   double a_1 = 2;
//   double a_2 = 3;
//   arma::vec delta = arma::zeros(5);
//   double alpha1 = 2;
//   double beta1 = 1;
//   double alpha2 = 3;
//   double beta2 = 1;
//   arma::mat A = arma::ones(1000, 2);
//   for(int i = 0; i < 1000; i++){
//     for(int j = 0; j < 5; j++){
//       if(j == 0){
//         delta(j) = R::rgamma(a_1, 1);
//       }else{
//         delta(j) = R::rgamma(a_2, 1);
//       }
//     }
//     BayesFPMM::updateA(alpha1, beta1, alpha2, beta2, delta, sqrt(1), sqrt(1), i, 1000, A);
//   }
//
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("a_1", a_1),
//                                       Rcpp::Named("a_2", a_2),
//                                       Rcpp::Named("A", A));
//   return mod;
// }
// //
// //' Tests updating Gamma
// //'
// //' @name TestUpdateGamma
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateGamma1(){
//   // Specify hyperparameters
//   double nu = 0.5;
//   // Make Delta vector
//   arma::vec Delta = {2,3};
//   // Make Gamma cube
//   arma::cube Gamma(3,8,2);
//   for(int i=0; i < 2; i++){
//     for(int j = 0; j < 3; j++){
//       for(int k = 0; k < 8; k++){
//         Gamma(j,k,i) =  R::rgamma(nu/2, 2/nu);
//       }
//     }
//   }
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,2);
//   arma::field<arma::cube> gamma(1000,1);
//   for(int i = 0; i < 1000; i++){
//     gamma(i,0) = arma::zeros(3,8,2);
//   }
//   for(int m = 0; m < 1000; m++){
//     double tau  = 1;
//     for(int i=0; i < 2; i++){
//       tau = tau * Delta(i);
//       for(int j=0; j < 3; j++){
//         for(int k=0; k < 8; k++){
//           Phi(j,k,i) = R::rnorm(0, (1/ std::pow(Gamma(j,k,i)*tau, 0.5)));
//         }
//       }
//     }
//     BayesFPMM::updateGamma(0.001, Delta, Phi, m, 250, gamma);
//   }
//   arma::vec gamma_ph = arma::zeros(1000);
//   arma::cube gamma_est = arma::zeros(3,8,2);
//   for(int i = 0; i < 3; i++){
//     for(int j = 0; j < 8; j++){
//       for(int l = 0; l < 2; l++){
//         for(int k = 0; k < 1000; k++){
//           gamma_ph(k) = gamma(k,0)(i,j,l);
//         }
//         gamma_est(i,j,l) = arma::median(gamma_ph);
//       }
//     }
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("gamma", Gamma),
//                                       Rcpp::Named("gamma_iter", gamma),
//                                       Rcpp::Named("gamma_est", gamma_est),
//                                       Rcpp::Named("delta", Delta));
//   return mod;
// }
//
// //' Tests updating Nu
// //'
// //' @name TestUpdateNu
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateNu(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++)
//   {
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.01;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   //Make Z
//   arma::mat Z(100, 3);
//   arma::vec c(3, arma::fill::randu);
//   arma::vec pi = rdirichlet(c);
//
//   // setting alpha_3 = 10
//   arma:: vec alpha = pi * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   arma::cube Nu_samp(nu.n_rows, nu.n_cols, 100);
//   arma::vec b_1(nu.n_cols, arma::fill::zeros);
//   arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   arma::mat P(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   P.zeros();
//   for(int j = 0; j < P.n_rows; j++){
//     P(0,0) = 1;
//     if(j > 0){
//       P(j,j) = 2;
//       P(j-1,j) = -1;
//       P(j,j-1) = -1;
//     }
//     P(P.n_rows - 1, P.n_rows - 1) = 1;
//   }
//   arma::vec tau(nu.n_rows, arma::fill::ones);
//   tau = tau / 10;
//   arma::vec log_lik = arma::zeros(100);
//   for(int i = 0; i < 100; i++){
//     updateNu(y_obs, B_obs, tau, Phi, Z, chi, sigma_sq, i, 100,
//              P, b_1, B_1, Nu_samp);
//     log_lik(i) = calcLikelihood(y_obs, B_obs, Nu_samp.slice(i),
//             Phi, Z, chi, sigma_sq);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu_samp", Nu_samp),
//                                       Rcpp::Named("nu", nu),
//                                       Rcpp::Named("log_lik", log_lik));
//   return mod;
// }
//
// //' Tests updating Tau
// //'
// //' @name TestUpdateTau
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateTau1(){
//   arma::vec tau = {1, 2, 2, 3, 5, 6};
//   arma::mat P(100, 100, arma::fill::zeros);
//   P.zeros();
//   for(int j = 0; j < P.n_rows; j++){
//     P(0,0) = 1;
//     if(j > 0){
//       P(j,j) = 2;
//       P(j-1,j) = -1;
//       P(j,j-1) = -1;
//     }
//     P(P.n_rows - 1, P.n_rows - 1) = 1;
//   }
//   arma::mat nu(6, 100, arma::fill::zeros);
//   arma::mat tau_samp(1000, 6, arma::fill::zeros);
//   arma::vec zeros_nu(100, arma::fill::zeros);
//   for(int i = 0; i < 1000; i++){
//     for(int j = 0; j < 6; j++){
//       nu.row(j) = arma::mvnrnd(zeros_nu, arma::pinv(P * tau(j))).t();
//     }
//     BayesFPMM::updateTau(1, 1, nu, i, 1000, P, tau_samp);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("tau_samp", tau_samp),
//                                       Rcpp::Named("tau", tau));
//   return mod;
// }
//
//
// //' Tests updating Sigma
// //'
// //' @name TestUpdateSigma
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateSigma1(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   arma::field<arma::mat> B_star(100,1);
//
//
//   for(int i = 0; i < 100; i++)
//   {
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.5;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   arma::mat Z(100, 3);
//   arma::vec c(3, arma::fill::randu);
//
//   // setting alpha_3 = 10
//   arma:: vec alpha = c * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//   double alpha_0 = 1;
//   double beta_0 = 1;
//   arma::vec sigma_samp(1000, arma::fill::zeros);
//   for(int i = 0; i < 1000; i++){
//     BayesFPMM::updateSigma(y_obs, B_obs, alpha_0, beta_0, nu, Phi, Z, chi,
//                 i, 1000, sigma_samp);
//   }
//   sigma_samp.save("/Users/nicholasmarco/Projects/BayesFPMM/inst/test-data/sigma.txt", arma::arma_ascii);
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("sigma_samp", sigma_samp),
//                                       Rcpp::Named("sigma", sigma_sq));
//   return mod;
// }
//
// //' Tests updating chi
// //'
// //' @name TestUpdateChi
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateChi(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   arma::field<arma::mat> B_star(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++){
//     Phi.slice(i) = (5-i) * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.0001;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   arma::mat Z(100, 3);
//   arma::vec c(3, arma::fill::randu);
//   arma::vec pi = rdirichlet(c);
//
//   // setting alpha_3 = 10
//   arma:: vec alpha = pi * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   arma::cube chi_samp(100, 5, 1000, arma::fill::zeros);
//   chi_samp.slice(0) = chi;
//   for(int i = 0; i < 1000; i++){
//     for(int j = 0; j < 100; j++){
//       mean = arma::zeros(8);
//       for(int l = 0; l < 3; l++){
//         mean = mean + Z(j,l) * nu.row(l).t();
//         for(int m = 0; m < Phi.n_slices; m++){
//           mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//         }
//       }
//       y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//         arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//     }
//     updateChi(y_obs, B_obs, Phi, nu, Z, sigma_sq, i, 1000,
//               chi_samp);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("chi_samp", chi_samp),
//                                       Rcpp::Named("chi", chi),
//                                       Rcpp::Named("Z", Z));
//   return mod;
// }
//
// //' Tests BFOC function
// //'
// //' @name GetStuff
// //' @export
// // [[Rcpp::export]]
// Rcpp::List GetStuff(double sigma_sq, const std::string dir, const int  n_funct){
//   arma::field<arma::vec> t_obs1(100,1);
//   for(int i = 0; i < n_funct; i++){
//     t_obs1(i,0) =  arma::regspace(0, 10, 990);
//   }
//
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   arma::mat nu;
//   nu.load(dir + "nu.txt");
//
//
//   // Make Phi matrix
//   arma::cube Phi;
//   Phi.load(dir + "Phi.txt");
//   // double sigma_sq = 0.005;
//
//   // Make chi matrix
//   arma::mat chi;
//   chi.load(dir + "chi.txt");
//
//
//   // Make Z matrix
//   arma::mat Z;
//   Z.load(dir + "Z.txt");
//
//   arma::field<arma::vec> y_obs(n_funct, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < n_funct; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < nu.n_rows; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) =arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("y", y_obs),
//                                       Rcpp::Named("B", B_obs),
//                                       Rcpp::Named("Phi_true", Phi),
//                                       Rcpp::Named("Z_true", Z),
//                                       Rcpp::Named("nu_true", nu));
//   return mod;
// }
//
//
//
// //' Tests updating Z
// //'
// //' @name TestUpdateZ
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateZTempered(const double beta)
// {
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   Z.col(0) = arma::vec(100, arma::fill::ones);
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {0.95, 0.5, 0.5};
//
//   // Initialize placeholder
//   arma::mat Z_ph = arma::zeros(100, 3);
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(100, 3, 100);
//   for(int i = 0; i < 100; i++)
//   {
//     updateZTempered(beta, y_obs,  B_obs, Phi, nu, chi, pi,
//             sigma_sq, 0.6, i, 100, Z_ph, Z_samp);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
//                                       Rcpp::Named("Z",Z),
//                                       Rcpp::Named("f_obs", y_obs));
//   return mod;
// }
//
// //' Tests updating Phi using temperature
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdatePhiTempered(const double beta)
// {
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   // arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   // Z.col(0) = arma::vec(100, arma::fill::ones);
//   arma::mat Z(100, 3);
//   arma::mat alpha(100,3, arma::fill::randu);
//   alpha = alpha * 100;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {0.95, 0.5, 0.5};
//
//   arma::field<arma::cube> Phi_samp(100, 1);
//   for(int i = 0; i < 100 ; i++){
//     Phi_samp(i,0) = arma::zeros(Phi.n_rows, Phi.n_cols, Phi.n_slices);
//   }
//   arma::vec m_1(Phi.n_cols, arma::fill::zeros);
//   arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
//   arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
//   gamma = gamma * 10;
//   arma::vec tilde_tau = {2, 2.5, 3, 5, 10};
//   for(int i = 0; i < 100; i++){
//     updatePhiTempered(beta, y_obs, B_obs, nu, gamma, tilde_tau, Z, chi,
//               sigma_sq, i, 100, m_1, M_1, Phi_samp);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Phi", Phi),
//                                       Rcpp::Named("Phi_samp", Phi_samp));
//   return mod;
// }
//
// //' Tests updating Nu
// //'
// //' @name TestUpdateNuTemperd
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateNuTempered(const double beta){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat.submat(0, 0, t_obs.n_elem - 1, 7);
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.01;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   // arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   // Z.col(0) = arma::vec(100, arma::fill::ones);
//   arma::mat Z(100, 3);
//   arma::mat alpha(100,3, arma::fill::randu);
//   alpha = alpha * 100;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {0.95, 0.5, 0.5};
//
//   arma::cube Nu_samp(nu.n_rows, nu.n_cols, 100);
//   arma::vec b_1(nu.n_cols, arma::fill::zeros);
//   arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   arma::mat P(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   P.zeros();
//   for(int j = 0; j < P.n_rows; j++){
//     P(0,0) = 1;
//     if(j > 0){
//       P(j,j) = 2;
//       P(j-1,j) = -1;
//       P(j,j-1) = -1;
//     }
//     P(P.n_rows - 1, P.n_rows - 1) = 1;
//   }
//   arma::vec tau(nu.n_rows, arma::fill::ones);
//   tau = tau * 10;
//   arma::vec log_lik = arma::zeros(100);
//   for(int i = 0; i < 100; i++){
//     updateNuTempered(beta, y_obs, B_obs, tau, Phi, Z, chi, sigma_sq, i, 100,
//              P, b_1, B_1, Nu_samp);
//     log_lik(i) = calcLikelihood(y_obs, B_obs, Nu_samp.slice(i),
//             Phi, Z, chi, sigma_sq);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("nu_samp", Nu_samp),
//                                       Rcpp::Named("nu", nu),
//                                       Rcpp::Named("log_lik", log_lik));
//   return mod;
// }
//
// //' Tests updating Sigma
// //'
// //' @name TestUpdateSigmaTempered
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateSigmaTempered(const double beta){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10,990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat { bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.5;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   // arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   // Z.col(0) = arma::vec(100, arma::fill::ones);
//   arma::mat Z(100, 3);
//   arma::mat alpha(100,3, arma::fill::randu);
//   alpha = alpha * 100;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//   double alpha_0 = 1;
//   double beta_0 = 1;
//   arma::vec sigma_samp(100, arma::fill::zeros);
//   for(int i = 0; i < 100; i++){
//     updateSigmaTempered(beta, y_obs, B_obs, alpha_0, beta_0, nu, Phi, Z, chi,
//                 i, 100, sigma_samp);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("sigma_samp", sigma_samp),
//                                       Rcpp::Named("sigma", sigma_sq));
//   return mod;
// }
//
// //' Tests updating chi
// //'
// //' @name TestUpdateChiTempered
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateChiTempered(const double beta){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++){
//     Phi.slice(i) = (5-i) * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   // arma::mat Z = arma::randi<arma::mat>(100, 3, arma::distr_param(0,1));
//   // Z.col(0) = arma::vec(100, arma::fill::ones);
//   arma::mat Z(100, 3);
//   arma::mat alpha(100,3, arma::fill::randu);
//   alpha = alpha * 100;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   arma::cube chi_samp(100, 5, 1000, arma::fill::zeros);
//   chi_samp.slice(0) = chi;
//   for(int i = 0; i < 1000; i++){
//     for(int j = 0; j < 100; j++){
//       mean = arma::zeros(8);
//       for(int l = 0; l < 3; l++){
//         mean = mean + Z(j,l) * nu.row(l).t();
//         for(int m = 0; m < Phi.n_slices; m++){
//           mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//         }
//       }
//       y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//         arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//     }
//     updateChiTempered(beta, y_obs, B_obs, Phi, nu, Z, sigma_sq, i, 1000,
//               chi_samp);
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("chi_samp", chi_samp),
//                                       Rcpp::Named("chi", chi),
//                                       Rcpp::Named("Z", Z));
//   return mod;
// }
//
// //' simulates parameters
// //'
// //' @name getparams
// //' @export
// // [[Rcpp::export]]
// void getparms(int n_funct){
//   int k = 2;
//   arma::field<arma::vec> t_obs1(n_funct,1);
//
//   for(int i = 0; i < n_funct; i++){
//     t_obs1(i,0) =  arma::regspace(0, 10, 990);
//   }
//
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(n_funct,1);
//
//   for(int i = 0; i < n_funct; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(k,8, arma::fill::randn);
//   arma::mat P_mat(8, 8, arma::fill::zeros);
//   P_mat.zeros();
//   for(int j = 0; j < P_mat.n_rows; j++){
//     P_mat(0,0) = 1;
//     if(j > 0){
//       P_mat(j,j) = 2;
//       P_mat(j-1,j) = -1;
//       P_mat(j,j-1) = -1;
//     }
//     P_mat(P_mat.n_rows - 1, P_mat.n_rows - 1) = 1;
//   }
//   arma::vec M = arma::zeros(8);
//   nu.row(0) = arma::mvnrnd(M, 4 * P_mat).t();
//   nu.row(1) = arma::mvnrnd(M, 4 * P_mat).t();
//
//   // Make Phi matrix
//   arma::cube Phi(k,8,3);
//   for(int i=0; i < 3; i++){
//     Phi.slice(i) = (3-i) * 0.1 * arma::randn<arma::mat>(k,8);
//   }
//
//   // Make chi matrix
//   arma::mat chi(n_funct, 3, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(n_funct, k);
//   arma::vec alpha(k, arma::fill::ones);
//   arma::vec alpha_i = {100, 1};
//   alpha = alpha * 0.5;
//   for(int i = 0; i < Z.n_rows; i++){
//     if(i < n_funct * 0.3){
//       Z.row(i) = rdirichlet(alpha_i).t();
//     }else if( i <  n_funct * 0.6){
//       alpha_i = {1, 100};
//       Z.row(i) = rdirichlet(alpha_i).t();
//     // }else if(i <  n_funct * 0.6){
//     //   alpha_i = {1, 1, 100};
//     //   Z.row(i) = rdirichlet(alpha_i).t();
//     }else{
//       Z.row(i) = rdirichlet(alpha).t();
//     }
//   }
//
//   //save parameters
//   nu.save("/Users/nicholasmarco/Projects/BayesFPMM/data/nu.txt", arma::arma_ascii);
//   chi.save("/Users/nicholasmarco/Projects/BayesFPMM/data/chi.txt", arma::arma_ascii);
//   Phi.save("/Users/nicholasmarco/Projects/BayesFPMM/data/Phi.txt", arma::arma_ascii);
//   Z.save("/Users/nicholasmarco/Projects/BayesFPMM/data/Z.txt", arma::arma_ascii);
// }
//
// //' Tests updating Z using partial membership model
// //'
// //' @name TestUpdateZ_PM
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateZ_PM(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(100, 3);
//   arma::mat alpha(100,3, arma::fill::randu);
//   alpha = alpha * 100;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {1, 1, 1};
//
//   // Initialize placeholder
//   arma::vec Z_ph = arma::zeros(3);
//
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(100, 3, 1000);
//   for(int i = 0; i < 100; i++){
//     Z_samp.slice(0).row(i) = rdirichlet(pi).t();
//   }
//   for(int i = 0; i < 1000; i++)
//   {
//     updateZ_PM(y_obs, B_obs, Phi, nu, chi, pi,
//             sigma_sq, i, 1000, 1.0, 1000, Z_ph, Z_samp);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
//                                       Rcpp::Named("Z",Z),
//                                       Rcpp::Named("f_obs", y_obs));
//   return mod;
//
// }
//
// //' Tests updating pi using partial membership model
// //'
// //' @name TestUpdateZ_PM
// //' @export
// // [[Rcpp::export]]
// arma::mat TestUpdatepi1(){
//   // Make Z matrix
//   arma::mat Z(100, 3);
//   arma::vec c(3, arma::fill::ones);
//   arma::vec pi = BayesFPMM::rdirichlet(c);
//
//   // setting alpha_3 = 100
//   arma:: vec alpha = pi * 100;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
//   }
//
//   // Initialize placeholder
//   arma::vec pi_ph = arma::zeros(3);
//
//
//   //Initialize Z_samp
//   arma::mat pi_samp = arma::ones(3, 500);
//
//   pi_samp.col(0) = BayesFPMM::rdirichlet(c);
//
//   for(int i = 0; i < 500; i++)
//   {
//     BayesFPMM::updatePi_PM(100, Z, c, i, 500, 1000, pi_ph, pi_samp);
//   }
//   pi_samp.save("/Users/nicholasmarco/Projects/BayesFPMM/inst/test-data/pi.txt", arma::arma_ascii);
//   arma::vec pi_est = arma::zeros(3);
//   arma::vec ph_pi = arma::zeros(300);
//   for(int j = 0; j < 3; j++){
//     for(int l = 200; l < 500; l++){
//       ph_pi(l - 200) = pi_samp(j,l);
//     }
//     pi_est(j) = arma::median(ph_pi);
//   }
//   // normalize
//   for(int j = 0; j < 3; j++){
//     pi_est = pi_est / arma::accu(pi_est);
//   }
//
//   arma::mat mod = arma::zeros(3, 2);
//   mod.col(0) = pi_est;
//   mod.col(1) = pi;
//   return mod;
// }

// //' Tests updating pi using partial membership model
// //'
// //' @name TestUpdateZ_PM
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdatealpha3(){
//
//   // Make Z matrix
//   arma::mat Z(100, 3);
//   arma::vec c(3, arma::fill::ones);
//   arma::vec pi = BayesFPMM::rdirichlet(c);
//
//   // setting alpha_3 = 10
//   arma:: vec alpha = pi * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
//   }
//
//   arma::vec alpha_3(10000, arma::fill::ones);
//   alpha_3 = alpha_3;
//
//   for(int i = 0; i < 10000; i++)
//   {
//     BayesFPMM::updateAlpha3(pi, 0.1, Z, i, 10000, 0.05, alpha_3);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("alpha3_samp", alpha_3),
//                                       Rcpp::Named("alpha3", alpha));
//   return mod;
// }
//
//
// //' Tests the full Bayesian Functional Partial Membership Model
// //'
// //' @name TestBFPMM
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestBFPMM(const int tot_mcmc_iters, const int r_stored_iters,
//                      const std::string directory, const double sigma_sq){
//   arma::field<arma::vec> t_obs1(100,1);
//   int n_funct = 100;
//   for(int i = 0; i < n_funct; i++){
//     t_obs1(i,0) =  arma::regspace(0, 10, 990);
//   }
//
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu;
//   nu.load("c:\\Projects\\BayesFPMM\\data\\nu.txt");
//
//
//   // Make Phi matrix
//   arma::cube Phi;
//   Phi.load("c:\\Projects\\BayesFPMM\\data\\Phi.txt");
//   // double sigma_sq = 0.005;
//
//   // Make chi matrix
//   arma::mat chi;
//   chi.load("c:\\Projects\\BayesFPMM\\data\\chi.txt");
//
//
//   // Make Z matrix
//   arma::mat Z;
//   Z.load("c:\\Projects\\BayesFPMM\\data\\Z.txt");
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < nu.n_rows; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//    arma::vec c = arma::ones(2);
//
//   // start MCMC sampling
//   Rcpp::List mod1 = BFPMM(y_obs, t_obs1, n_funct, 50, 2, 8, 3, tot_mcmc_iters,
//                           r_stored_iters, c, 1, 3, 2, 3, 1, 1,
//                           1000, 1000, 0.05, sqrt(1), sqrt(1), 1, 1, 1, 1,
//                           directory);
//
//   Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
//                                         Rcpp::Named("y_obs", y_obs),
//                                         Rcpp::Named("nu_true", nu),
//                                         Rcpp::Named("Phi_true", Phi),
//                                         Rcpp::Named("nu", mod1["nu"]),
//                                         Rcpp::Named("chi", mod1["chi"]),
//                                         Rcpp::Named("pi", mod1["pi"]),
//                                         Rcpp::Named("alpha_3", mod1["alpha_3"]),
//                                         Rcpp::Named("A", mod1["A"]),
//                                         Rcpp::Named("delta", mod1["delta"]),
//                                         Rcpp::Named("sigma", mod1["sigma"]),
//                                         Rcpp::Named("tau", mod1["tau"]),
//                                         Rcpp::Named("gamma", mod1["gamma"]),
//                                         Rcpp::Named("Phi", mod1["Phi"]),
//                                         Rcpp::Named("Z", mod1["Z"]),
//                                         Rcpp::Named("loglik", mod1["loglik"]));
//
//   return mod2;
// }
//
// //' Tests updating Z using partial membership model
// //'
// //' @name TestUpdateZ_PM
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateTemperedZ_PM(double beta){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(100, 3);
//   arma::mat alpha(100,3, arma::fill::randu);
//   alpha = alpha * 100;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {1, 1, 1};
//
//   // Initialize placeholder
//   arma::vec Z_ph = arma::zeros(3);
//
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(100, 3, 1000);
//   for(int i = 0; i < 100; i++){
//     Z_samp.slice(0).row(i) = rdirichlet(pi).t();
//   }
//
//   for(int i = 0; i < 1000; i++)
//   {
//
//     updateZTempered_PM(beta, y_obs, B_obs, Phi, nu, chi, pi,
//                sigma_sq, i, 1000, 1.0, 1000, Z_ph, Z_samp);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
//                                       Rcpp::Named("Z",Z),
//                                       Rcpp::Named("f_obs", y_obs));
//   return mod;
// }
//
// //' Tests BFOC function
// //'
// //' @name TestEstimateBFPMMTempladder
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestEstimateBFPMMTempladder(const double beta_N_t, const int N_t){
//   arma::field<arma::vec> t_obs1(100,1);
//   int n_funct = 100;
//   for(int i = 0; i < n_funct; i++){
//     t_obs1(i,0) =  arma::regspace(0, 10, 990);
//   }
//
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   arma::field<arma::mat> B_star(100,1);
//
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu;
//   nu.load("c:\\Projects\\BayesFPMM\\data\\nu.txt");
//
//
//   // Make Phi matrix
//   arma::cube Phi;
//   Phi.load("c:\\Projects\\BayesFPMM\\data\\Phi.txt");
//   // double sigma_sq = 0.005;
//
//   // Make chi matrix
//   arma::mat chi;
//   chi.load("c:\\Projects\\BayesFPMM\\data\\chi.txt");
//
//
//   // Make Z matrix
//   arma::mat Z;
//   Z.load("c:\\Projects\\BayesFPMM\\data\\Z.txt");
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   double sigma_sq = 0.01;
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < nu.n_rows; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   arma::vec c = arma::ones(2);
//
//   int tot_mcmc_iters = 10;
//   int r_stored_iters = 10;
//
//   // start MCMC sampling
//   Rcpp::List mod1 = BFPMM_Templadder(y_obs, t_obs1, n_funct, 2, 8, 3, tot_mcmc_iters,
//                                      r_stored_iters, c, 1, 3, 2, 3, 1, 1,
//                                      1000, 1000, 0.05, sqrt(1), sqrt(1), 1, 1, 1, 1, beta_N_t,
//                                      N_t);
//
//   Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
//                                         Rcpp::Named("alpha_3", mod1["alpha_3"]),
//                                         Rcpp::Named("y_obs", y_obs),
//                                         Rcpp::Named("nu_true", nu),
//                                         Rcpp::Named("Phi_true", Phi),
//                                         Rcpp::Named("nu", mod1["nu"]),
//                                         Rcpp::Named("chi", mod1["chi"]),
//                                         Rcpp::Named("pi", mod1["pi"]),
//                                         Rcpp::Named("A", mod1["A"]),
//                                         Rcpp::Named("delta", mod1["delta"]),
//                                         Rcpp::Named("sigma", mod1["sigma"]),
//                                         Rcpp::Named("tau", mod1["tau"]),
//                                         Rcpp::Named("gamma", mod1["gamma"]),
//                                         Rcpp::Named("Phi", mod1["Phi"]),
//                                         Rcpp::Named("Z", mod1["Z"]));
//
//   return mod2;
// }
//
// //' Tests the full Bayesian Functional Partial Membership Model
// //'
// //' @name TestBFPMM_Nu_Z
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestBFPMM_Nu_Z(const int tot_mcmc_iters, const double sigma_sq,
//                           const double beta_N_t, const int N_t, const int n_temp_trans){
//   arma::field<arma::vec> t_obs1(100,1);
//   int n_funct = 100;
//   for(int i = 0; i < n_funct; i++){
//     t_obs1(i,0) =  arma::regspace(0, 10, 990);
//   }
//
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   arma::vec internal_knots = {250, 500, 750};
//   arma::vec boundary_knots = {0, 990};
//   int basis_degree = 3;
//
//   bspline = splines2::BSpline(t_obs, internal_knots, basis_degree,
//                               boundary_knots);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat { bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(100,1);
//
//   for(int i = 0; i < 100; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu;
//   nu.load("c:\\Projects\\BayesFPMM\\data\\nu.txt");
//
//
//   // Make Phi matrix
//   arma::cube Phi;
//   Phi.load("c:\\Projects\\BayesFPMM\\data\\Phi.txt");
//   // double sigma_sq = 0.005;
//
//   // Make chi matrix
//   arma::mat chi;
//   chi.load("c:\\Projects\\BayesFPMM\\data\\chi.txt");
//
//
//   // Make Z matrix
//   arma::mat Z;
//   Z.load("c:\\Projects\\BayesFPMM\\data\\Z.txt");
//
//   arma::field<arma::vec> y_obs(100, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < nu.n_rows; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//   arma::vec c = arma::ones(2);
//
//   // start MCMC sampling
//   Rcpp::List mod1 = BFPMM_Nu_Z(y_obs, t_obs1, n_funct, 2, 3, 3,
//                                boundary_knots, internal_knots, tot_mcmc_iters,
//                                c, 1, 3, 2, 3, 1, 1, 1000, 1000,
//                                0.05, sqrt(1), sqrt(1), 1, 1, 1, 1);
//
//   Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
//                                         Rcpp::Named("y_obs", y_obs),
//                                         Rcpp::Named("nu_true", nu),
//                                         Rcpp::Named("nu", mod1["nu"]),
//                                         Rcpp::Named("pi", mod1["pi"]),
//                                         Rcpp::Named("alpha_3", mod1["alpha_3"]),
//                                         Rcpp::Named("A", mod1["A"]),
//                                         Rcpp::Named("delta", mod1["delta"]),
//                                         Rcpp::Named("sigma", mod1["sigma"]),
//                                         Rcpp::Named("tau", mod1["tau"]),
//                                         Rcpp::Named("Z", mod1["Z"]),
//                                         Rcpp::Named("loglik", mod1["loglik"]));
//
//   return mod2;
// }
//
//
// //' Tests the full Bayesian Functional Partial Membership Model
// //'
// //' @name TestBFPMM_Theta
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestBFPMM_Theta(const int tot_mcmc_iters, const double sigma_sq,
//                            const arma::cube Z_samp, const arma::cube nu_samp,
//                            double burnin_prop, const int k,const std::string dir){
//   // Make Z matrix
//   arma::mat Z;
//   Z.load(dir + "Z.txt");
//   int n_funct = Z.n_rows;
//
//   arma::field<arma::vec> t_obs1(n_funct,1);
//
//   for(int i = 0; i < n_funct; i++){
//     t_obs1(i,0) =  arma::regspace(0, 10, 990);
//   }
//
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   arma::vec internal_knots = {250, 500, 750};
//   arma::vec boundary_knots = {0, 990};
//   int basis_degree = 3;
//
//   bspline = splines2::BSpline(t_obs, internal_knots, basis_degree,
//                               boundary_knots);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(n_funct,1);
//
//   for(int i = 0; i < n_funct; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu;
//   nu.load(dir + "nu.txt");
//
//
//   // Make Phi matrix
//   arma::cube Phi;
//   Phi.load(dir + "Phi.txt");
//   // double sigma_sq = 0.005;
//
//   // Make chi matrix
//   arma::mat chi;
//   chi.load(dir + "chi.txt");
//
//
//   arma::field<arma::vec> y_obs(n_funct, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < n_funct; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < nu.n_rows; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//   arma::vec c = arma::ones(k);
//
//   int n_nu = nu_samp.n_slices;
//   arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
//   arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
//   arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
//   arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
//   for(int i = 0; i < Z_est.n_cols; i++){
//     for(int j = 0; j < Z_est.n_rows; j++){
//       for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
//         ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
//       }
//       Z_est(j,i) = arma::median(ph_Z);
//     }
//     for(int j = 0; j < nu_samp.n_cols; j++){
//       for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
//         ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
//       }
//       nu_est(i,j) = arma::median(ph_nu);
//     }
//   }
//
//   // rescale Z and nu
//   arma::mat transform_mat = arma::zeros(Z_samp.n_cols, Z_samp.n_cols);
//   int max_ind = 0;
//   for(int i = 0; i < Z_est.n_cols; i++){
//     max_ind = arma::index_max(Z_est.col(i));
//     transform_mat.row(i) = Z_est.row(max_ind);
//   }
//   arma::mat Z_est_rescale = Z_est * arma::pinv(transform_mat);
//   arma::mat nu_est_rescale = arma::pinv(transform_mat) * nu_est;
//   Z_est_rescale(arma::index_max(Z_est_rescale.col(0)),0) = 0.999;
//   Z_est_rescale(arma::index_max(Z_est_rescale.col(0)),1) = 0.001;
//   Z_est_rescale(arma::index_max(Z_est_rescale.col(1)),0) = 0.001;
//   Z_est_rescale(arma::index_max(Z_est_rescale.col(1)),1) = 0.999;
//
//
//   // arma::mat Z_est_rescale = arma::zeros(n_funct, Z_samp.n_cols);
//   // arma::mat nu_est_rescale = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
//   // double Z_max = 0;
//   // double Z_min = 0;
//   // int Z_max_ind = 0;
//   // int Z_min_ind = 0;
//   // for(int i = 0; i < Z_est.n_cols; i++){
//   //   Z_max = arma::max(Z.col(i));
//   //   Z_min = arma::min(Z.col(i));
//   //   Z_max_ind = arma::index_max(Z.col(i));
//   //   for(int j = 0; j < Z_est.n_rows; j++){
//   //     Z_est_rescale(i,j) = (Z_est(i,j) - Z_min) / (Z_max - Z_min);
//   //     for(int l = 0; l < Z_est.n_cols; l++){
//   //       nu_est_rescale.row(i) = nu_est_rescale.row(i) + Z_est(Z_max_ind, l) * nu_est.row(l);
//   //     }
//   //   }
//   // }
//
//   // start MCMC sampling
//   Rcpp::List mod1 = BFPMM_Theta(y_obs, t_obs1, n_funct, k, 3, 3, boundary_knots,
//                                 internal_knots, tot_mcmc_iters,
//                                 c, 1, 3, 2, 3, 1, 1, 1000, 1000, 0.05,
//                                 sqrt(1), sqrt(1), 1, 5, 1, 1, Z_est_rescale, nu_est_rescale);
//
//   Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("chi", mod1["chi"]),
//                                         Rcpp::Named("Z", mod1["Z"]),
//                                         Rcpp::Named("nu", mod1["nu"]),
//                                         Rcpp::Named("A", mod1["A"]),
//                                         Rcpp::Named("delta", mod1["delta"]),
//                                         Rcpp::Named("sigma", mod1["sigma"]),
//                                         Rcpp::Named("tau", mod1["tau"]),
//                                         Rcpp::Named("gamma", mod1["gamma"]),
//                                         Rcpp::Named("Phi", mod1["Phi"]),
//                                         Rcpp::Named("Nu_est", nu_est),
//                                         Rcpp::Named("loglik", mod1["loglik"]));
//
//   return mod2;
// }
//
//
// //' Tests the full Bayesian Functional Partial Membership Model
// //'
// //' @name TestBFPMM_Nu_Z
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestBFPMM_Nu_Z_multiple_try(const int tot_mcmc_iters, const double sigma_sq,
//                                        const double beta_N_t, const int N_t, const int n_temp_trans,
//                                        const int n_trys, const int k, const std::string dir){
//   // Make Z matrix
//   arma::mat Z;
//   Z.load(dir + "Z.txt");
//   int n_funct = Z.n_rows;
//
//   arma::field<arma::vec> t_obs1(n_funct,1);
//   for(int i = 0; i < n_funct; i++){
//     t_obs1(i,0) =  arma::regspace(0, 10, 990);
//   }
//
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   arma::vec internal_knots = {250, 500, 750};
//   arma::vec boundary_knots = {0, 990};
//   int basis_degree = 3;
//
//   bspline = splines2::BSpline(t_obs, internal_knots, basis_degree,
//                               boundary_knots);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(n_funct,1);
//
//   for(int i = 0; i < n_funct; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu;
//   nu.load(dir + "nu.txt");
//
//
//   // Make Phi matrix
//   arma::cube Phi;
//   Phi.load(dir + "Phi.txt");
//   // double sigma_sq = 0.005;
//
//   // Make chi matrix
//   arma::mat chi;
//   chi.load(dir + "chi.txt");
//
//
//   arma::field<arma::vec> y_obs(n_funct, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < n_funct; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < nu.n_rows; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//   arma::vec c = arma::ones(k);
//
//   // start MCMC sampling
//   Rcpp::List mod1 = BFPMM_Nu_Z(y_obs, t_obs1, n_funct, 2, 3, 3,
//                                boundary_knots, internal_knots, tot_mcmc_iters,
//                                c, 1, 3, 2, 3, 1, 1, 1000, 1000,
//                                0.05, sqrt(1), sqrt(1), 1, 1, 1, 1);
//   arma::vec ph = mod1["loglik"];
//   double min_likelihood = arma::mean(ph.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
//
//   for(int i = 0; i < n_trys; i++){
//     Rcpp::List modi = BFPMM_Nu_Z(y_obs, t_obs1, n_funct, 2, 3, 3,
//                                  boundary_knots, internal_knots, tot_mcmc_iters,
//                                  c, 1, 3, 2, 3, 1, 1, 1000, 1000,
//                                  0.05, sqrt(1), sqrt(1), 1, 1, 1, 1);
//     arma::vec ph1 = modi["loglik"];
//     if(min_likelihood < arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1))){
//       mod1 = modi;
//       min_likelihood = arma::mean(ph1.subvec((tot_mcmc_iters)-99, (tot_mcmc_iters)-1));
//     }
//
//   }
//
//   Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
//                                         Rcpp::Named("y_obs", y_obs),
//                                         Rcpp::Named("nu_true", nu),
//                                         Rcpp::Named("nu", mod1["nu"]),
//                                         Rcpp::Named("pi", mod1["pi"]),
//                                         Rcpp::Named("alpha_3", mod1["alpha_3"]),
//                                         Rcpp::Named("A", mod1["A"]),
//                                         Rcpp::Named("delta", mod1["delta"]),
//                                         Rcpp::Named("sigma", mod1["sigma"]),
//                                         Rcpp::Named("tau", mod1["tau"]),
//                                         Rcpp::Named("Z", mod1["Z"]),
//                                         Rcpp::Named("loglik", mod1["loglik"]));
//
//   return mod2;
// }
//
// //' Tests mixed sampling from the Bayesian Functional Partial Membership Model
// //'
// //' @name TestBFPMM_MTT
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestBFPMM_MTT_warm_start(const double beta_N_t,
//                                     const int N_t,
//                                     const int n_temp_trans,
//                                     const int tot_mcmc_iters,
//                                     const int r_stored_iters,
//                                     const std::string directory,
//                                     const double sigma_sq,
//                                     const arma::cube Z_samp,
//                                     const arma::mat pi_samp,
//                                     const arma::vec alpha_3_samp,
//                                     const arma::mat delta_samp,
//                                     const arma::field<arma::cube> gamma_samp,
//                                     const arma::field<arma::cube> Phi_samp,
//                                     const arma::mat A_samp,
//                                     const arma::cube nu_samp,
//                                     const arma::mat tau_samp,
//                                     const arma::vec sigma_samp,
//                                     const arma::cube chi_samp,
//                                     const double burnin_prop,
//                                     const int k,
//                                     const std::string dir){
//
//   // Make Z matrix
//   arma::mat Z;
//   Z.load(dir + "Z.txt");
//   int n_funct = Z.n_rows;
//
//   arma::field<arma::vec> t_obs1(n_funct,1);
//
//   for(int i = 0; i < n_funct; i++){
//     t_obs1(i,0) =  arma::regspace(0, 10, 990);
//   }
//
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//
//   arma::vec internal_knots = {250, 500, 750};
//   arma::vec boundary_knots = {0, 990};
//   int basis_degree = 3;
//
//   bspline = splines2::BSpline(t_obs, internal_knots, basis_degree,
//                               boundary_knots);
//
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(n_funct,1);
//
//   for(int i = 0; i < n_funct; i++)
//   {
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu;
//   nu.load(dir + "nu.txt");
//
//
//   // Make Phi matrix
//   arma::cube Phi;
//   Phi.load(dir + "Phi.txt");
//   // double sigma_sq = 0.005;
//
//   // Make chi matrix
//   arma::mat chi;
//   chi.load(dir + "chi.txt");
//
//   arma::field<arma::vec> y_obs(n_funct, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < n_funct; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < nu.n_rows; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//   arma::vec c = arma::ones(k);
//   c = c / 5;
//
//   int n_nu = alpha_3_samp.n_elem;
//
//   double alpha_3_est = arma::median(alpha_3_samp.subvec(std::round(n_nu * burnin_prop), n_nu - 1));
//   arma::vec pi_est = arma::zeros(pi_samp.n_rows);
//   arma::mat Z_est = arma::zeros(n_funct, Z_samp.n_cols);
//   arma::mat nu_est = arma::zeros(nu_samp.n_rows, nu_samp.n_cols);
//   arma::vec ph_Z = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
//   arma::vec ph_nu = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
//   for(int i = 0; i < Z_est.n_cols; i++){
//     pi_est(i) = arma::median(pi_samp.row(i).subvec(std::round(n_nu * burnin_prop), n_nu - 1));
//     for(int j = 0; j < Z_est.n_rows; j++){
//       for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
//         ph_Z(l - std::round(n_nu * burnin_prop)) = Z_samp(j,i,l);
//       }
//       Z_est(j,i) = arma::median(ph_Z);
//     }
//     for(int j = 0; j < nu_samp.n_cols; j++){
//       for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
//         ph_nu(l - std::round(n_nu * burnin_prop)) = nu_samp(i,j,l);
//       }
//       nu_est(i,j) = arma::median(ph_nu);
//     }
//   }
//
//   int n_Phi = sigma_samp.n_elem;
//
//   double sigma_est = arma::median(sigma_samp.subvec(std::round(n_Phi * burnin_prop), n_Phi - 1));
//   arma::vec delta_est = arma::zeros(delta_samp.n_rows);
//   arma::vec ph_delta = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
//   for(int i = 0; i < delta_samp.n_rows; i++){
//     for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
//       ph_delta(l - std::round(n_Phi * burnin_prop)) = delta_samp(i,l);
//     }
//     delta_est(i) = arma::median(ph_delta);
//   }
//   arma::cube gamma_est = arma::zeros(gamma_samp(0,0).n_rows, gamma_samp(0,0).n_cols, gamma_samp(0,0).n_slices);
//   arma::cube Phi_est = arma::zeros(Phi_samp(0,0).n_rows, Phi_samp(0,0).n_cols, Phi_samp(0,0).n_slices);
//   arma::vec ph_phi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
//   arma::vec ph_gamma = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
//   for(int i = 0; i < Phi_est.n_rows; i++){
//     for(int j = 0; j < Phi_est.n_cols; j++){
//       for(int m = 0; m < Phi_est.n_slices; m++){
//         for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
//           ph_phi(l - std::round(n_Phi * burnin_prop)) = Phi_samp(l,0)(i,j,m);
//
//           ph_gamma(l - std::round(n_Phi * burnin_prop)) = gamma_samp(l,0)(i,j,m);
//         }
//         Phi_est(i,j,m) = arma::median(ph_phi);
//         gamma_est(i,j,m) = arma::median(ph_gamma);
//       }
//     }
//   }
//
//   arma::vec A_est = arma::zeros(A_samp.n_cols);
//   arma::vec ph_A = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
//   for(int i = 0; i < A_est.n_elem; i++){
//     for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
//       ph_A(l - std::round(n_Phi * burnin_prop)) = A_samp(l,i);
//     }
//     A_est(i) = arma::median(ph_A);
//   }
//   arma::vec tau_est = arma::zeros(tau_samp.n_cols);
//   arma::vec ph_tau = arma::zeros(n_nu - std::round(n_nu * burnin_prop));
//   for(int i = 0; i < tau_est.n_elem; i++){
//     for(int l = std::round(n_nu * burnin_prop); l < n_nu; l++){
//       ph_tau(l - std::round(n_nu * burnin_prop)) = tau_samp(l,i);
//     }
//     tau_est(i) = arma::median(ph_tau);
//   }
//   arma::mat chi_est = arma::zeros(chi_samp.n_rows, chi_samp.n_cols);
//   arma::vec ph_chi = arma::zeros(n_Phi - std::round(n_Phi * burnin_prop));
//   for(int i = 0; i < chi_est.n_rows; i++){
//     for(int j = 0; j < chi_est.n_cols; j++){
//       for(int l = std::round(n_Phi * burnin_prop); l < n_Phi; l++){
//         ph_chi(l - std::round(n_Phi * burnin_prop)) = chi_samp(i,j,l);
//       }
//       chi_est(i,j) = arma::median(ph_chi);
//     }
//   }
//
//   // start MCMC sampling
//   Rcpp::List mod1 = BFPMM_MTT_warm_start(y_obs, t_obs1, n_funct, 50, k, 3, 3,
//                                          boundary_knots, internal_knots, tot_mcmc_iters,
//                                          r_stored_iters, n_temp_trans, c, 1, 3, 2,
//                                          3, 1, 1, 1000, 1000, 0.05, sqrt(1), sqrt(1), 1, 10, 1, 1,
//                                          directory, beta_N_t, N_t, Z_est, pi_est, alpha_3_est,
//                                          delta_est, gamma_est, Phi_est, A_est, nu_est,
//                                          tau_est, sigma_est, chi_est);
//
//   Rcpp::List mod2 =  Rcpp::List::create(Rcpp::Named("Z_true", Z),
//                                         Rcpp::Named("y_obs", y_obs),
//                                         Rcpp::Named("nu_true", nu),
//                                         Rcpp::Named("Phi_true", Phi),
//                                         Rcpp::Named("nu", mod1["nu"]),
//                                         Rcpp::Named("chi", mod1["chi"]),
//                                         Rcpp::Named("pi", mod1["pi"]),
//                                         Rcpp::Named("alpha_3", mod1["alpha_3"]),
//                                         Rcpp::Named("A", mod1["A"]),
//                                         Rcpp::Named("delta", mod1["delta"]),
//                                         Rcpp::Named("sigma", mod1["sigma"]),
//                                         Rcpp::Named("tau", mod1["tau"]),
//                                         Rcpp::Named("gamma", mod1["gamma"]),
//                                         Rcpp::Named("Phi", mod1["Phi"]),
//                                         Rcpp::Named("Z", mod1["Z"]),
//                                         Rcpp::Named("loglik", mod1["loglik"]));
//
//   return mod2;
// }
//
// //' Tests update of Z for multivariate model
// //'
// //' @name TestUpdateZ_MV
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateZ_MV(){
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(100, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(100, 3);
//   arma::mat alpha(100,3, arma::fill::randu);
//   alpha = alpha * 100;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::mat y_obs = arma::zeros(100, 8);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs.row(j) = arma::mvnrnd(mean, sigma_sq *
//       arma::eye(mean.n_elem, mean.n_elem)).t();
//   }
//
//   // Initialize pi
//   arma::vec pi = {1, 1, 1};
//
//   // Initialize placeholder
//   arma::vec Z_ph = arma::zeros(3);
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(100, 3, 1000);
//   for(int i = 0; i < 100; i++){
//     Z_samp.slice(0).row(i) = rdirichlet(pi).t();
//   }
//   for(int i = 0; i < 1000; i++)
//   {
//     updateZ_PMMV(y_obs, Phi, nu, chi, pi,
//                sigma_sq, i, 1000, 1.0, 1000, Z_ph, Z_samp);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Z_samp", Z_samp),
//                                       Rcpp::Named("Z",Z),
//                                       Rcpp::Named("f_obs", y_obs));
//   return mod;
// }
//
//


// //' Tests updating Z using partial membership model
// //'
// //' @name TestUpdateZ_PM
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateZ_PM(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(20,1);
//
//   for(int i = 0; i < 20; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.0001;
//
//   // Make chi matrix
//   arma::mat chi(20, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(20, 3);
//   arma::mat alpha(20,3, arma::fill::ones);
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(20, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 20; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {1, 1, 1};
//
//   // Initialize placeholder
//   arma::vec Z_ph = arma::zeros(3);
//
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(20, 3, 5000);
//   for(int i = 0; i < 20; i++){
//     Z_samp.slice(0).row(i) = BayesFPMM::rdirichlet(pi).t();
//   }
//   for(int i = 0; i < 5000; i++){
//     BayesFPMM::updateZ_PM(y_obs, B_obs, Phi, nu, chi, pi,
//                           sigma_sq, i, 5000, 1.0, 2000, Z_ph, Z_samp);
//   }
//   arma::mat Z_est = arma::zeros(20, 3);
//   arma::vec ph_Z = arma::zeros(500);
//   for(int i = 0; i < 20; i++){
//     for(int j = 0; j < 3; j++){
//       for(int l = 4500; l < 5000; l++){
//         ph_Z(l - 4500) = Z_samp(i,j,l);
//       }
//       Z_est(i,j) = arma::median(ph_Z);
//     }
//   }
//
//   Rcpp::List mod= Rcpp::List::create(Rcpp::Named("true", Z),
//                                      Rcpp::Named("est", Z_est),
//                                      Rcpp::Named("chain", Z_samp));
//   return mod;
// }

// //' Tests updating Z using partial membership model
// //'
// //' @name TestUpdateZ_PM
// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdateTemperedZ_PM(double beta){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(20,1);
//
//   for(int i = 0; i < 20; i++){
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.0001;
//
//   // Make chi matrix
//   arma::mat chi(20, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(20, 3);
//   arma::mat alpha(20,3, arma::fill::ones);
//   alpha = alpha * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(20, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 20; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {1, 1, 1};
//
//   // Initialize placeholder
//   arma::vec Z_ph = arma::zeros(3);
//
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(20, 3, 1000);
//   for(int i = 0; i < 20; i++){
//     Z_samp.slice(0).row(i) = BayesFPMM::rdirichlet(pi).t();
//   }
//
//   for(int i = 0; i < 1000; i++){
//     BayesFPMM::updateZTempered_PM(beta, y_obs, B_obs, Phi, nu, chi, pi,
//                                   sigma_sq, i, 1000, 1.0, 2000, Z_ph, Z_samp);
//   }
//
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("true", Z),
//                                      Rcpp::Named("est", Z_samp));
//   return mod;
// }

// //' @export
// // [[Rcpp::export]]
// arma::cube TestUpdateTemperedZ_MV(){
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.2 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(20, 5, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(20, 3);
//   arma::mat alpha(20,3, arma::fill::ones);
//   alpha = alpha * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::mat y_obs = arma::zeros(20, 8);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 20; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs.row(j) = arma::mvnrnd(mean, sigma_sq *
//       arma::eye(mean.n_elem, mean.n_elem)).t();
//   }
//
//   // Initialize pi
//   arma::vec pi = {10, 10, 10};
//
//   // Initialize placeholder
//   arma::vec Z_ph = arma::zeros(3);
//
//   //Initialize Z_samp
//   arma::cube Z_samp = arma::ones(20, 3, 500);
//   for(int i = 0; i < 20; i++){
//     Z_samp.slice(0).row(i) = BayesFPMM::rdirichlet(pi).t();
//   }
//   double beta = 0.05;
//
//   for(int i = 0; i < 500; i++)
//   {
//     BayesFPMM::updateZTempered_PMMV(beta, y_obs, Phi, nu, chi, pi,
//                                     sigma_sq, i, 500, 1.0, 2000, Z_ph, Z_samp);
//   }
//   arma::mat Z_est = arma::zeros(20, 3);
//   arma::vec ph_Z = arma::zeros(500);
//   for(int i = 0; i < 20; i++){
//     for(int j = 0; j < 3; j++){
//       for(int l = 200; l < 500; l++){
//         ph_Z(l - 200) = Z_samp(i,j,l);
//       }
//       Z_est(i,j) = arma::median(ph_Z);
//     }
//   }
//   // normalize
//   for(int i = 0; i < 20; i++){
//     for(int j = 0; j < 3; j++){
//       Z_est.row(i) = Z_est.row(i) / arma::accu(Z_est.row(i));
//     }
//   }
//
//   arma::cube mod = arma::zeros(20, 3, 2);
//   mod.slice(0) = Z_est;
//   mod.slice(1) = Z;
//   return mod;
// }

// //' @export
// // [[Rcpp::export]]
// arma::cube TestUpdateNuTem(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (20 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(20,1);
//
//   for(int i = 0; i < 20; i++)
//   {
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.01;
//
//   // Make chi matrix
//   arma::mat chi(20, 5, arma::fill::randn);
//
//
//   //Make Z
//   arma::mat Z(20, 3);
//   arma::vec c(3, arma::fill::ones);
//   arma::vec pi = BayesFPMM::rdirichlet(c);
//
//   // setting alpha_3 = 10
//   arma:: vec alpha = pi * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
//   }
//
//   arma::field<arma::vec> y_obs(20, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 20; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   arma::cube Nu_samp = arma::randn(nu.n_rows, nu.n_cols, 500);
//   arma::vec b_1(nu.n_cols, arma::fill::zeros);
//   arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   arma::mat P(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   P.zeros();
//   for(int j = 0; j < P.n_rows; j++){
//     P(0,0) = 1;
//     if(j > 0){
//       P(j,j) = 2;
//       P(j-1,j) = -1;
//       P(j,j-1) = -1;
//     }
//     P(P.n_rows - 1, P.n_rows - 1) = 1;
//   }
//   arma::vec tau(nu.n_rows, arma::fill::ones);
//   tau = tau / 10;
//   for(int i = 0; i < 500; i++){
//     BayesFPMM::updateNuTempered(0.4, y_obs, B_obs, tau, Phi, Z, chi, sigma_sq, i, 500,
//              P, b_1, B_1, Nu_samp);
//   }
//
//   arma::cube mod = arma::zeros(3,8,2);
//   arma::vec nu_ph = arma::zeros(200);
//   arma::mat nu_est = arma::zeros(3,8);
//   for(int i = 0; i < 3; i++){
//     for(int j = 0; j < 8; j++){
//       for(int k = 300; k < 500; k++){
//         nu_ph(k - 300) = Nu_samp(i,j,k);
//       }
//       nu_est(i,j) = arma::median(nu_ph);
//     }
//   }
//   mod.slice(1) = nu;
//   mod.slice(0) = nu_est;
//   return mod;
// }


// //' @export
// // [[Rcpp::export]]
// Rcpp::List TestUpdatePhi1()
// {
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (100 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(40,1);
//
//
//   for(int i = 0; i < 40; i++)
//   {
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,2);
//   for(int i=0; i < 2; i++)
//   {
//     Phi.slice(i) = (2-i) * arma::randn<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(40, 2, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(40, 3);
//   arma::mat alpha(40,3, arma::fill::ones);
//   alpha = alpha * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::field<arma::vec> y_obs(40, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 40; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   // Initialize pi
//   arma::vec pi = {0.5, 0.5, 0.5};
//
//   arma::field<arma::cube> Phi_samp(250, 1);
//   for(int i = 0; i < 250 ; i++){
//     Phi_samp(i,0) = arma::randn(Phi.n_rows, Phi.n_cols, Phi.n_slices);
//   }
//   arma::vec m_1(Phi.n_cols, arma::fill::zeros);
//   arma::mat M_1(Phi.n_cols, Phi.n_cols, arma::fill::zeros);
//   arma::cube gamma(Phi.n_rows, Phi.n_cols, Phi.n_slices, arma::fill::ones);
//   gamma = gamma * 10;
//   arma::vec tilde_tau = {1, 2};
//   for(int i = 0; i < 250; i++){
//     BayesFPMM::updatePhi(y_obs, B_obs, nu, gamma, tilde_tau, Z, chi,
//               sigma_sq, i, 250, m_1, M_1, Phi_samp);
//   }
//   Phi_samp.save("/Users/nicholasmarco/Projects/BayesFPMM/inst/test-data/Phi.txt");
//   arma::vec phi_ph = arma::zeros(150);
//   arma::cube phi_est = arma::zeros(3,8,2);
//   for(int i = 0; i < 3; i++){
//     for(int j = 0; j < 8; j++){
//       for(int l = 0; l < 2; l++){
//         for(int k = 100; k < 250; k++){
//           phi_ph(k - 100) = Phi_samp(k,0)(i,j,l);
//         }
//         phi_est(i,j,l) = arma::median(phi_ph);
//       }
//     }
//   }
//   Rcpp::List mod = Rcpp::List::create(Rcpp::Named("Phi", Phi),
//                                       Rcpp::Named("Phi_samp", Phi_samp),
//                                       Rcpp::Named("Phi_est", phi_est));
//   return mod;
// }

// //' Tests updating Nu
// //'
// //' @name TestUpdateNu
// //' @export
// // [[Rcpp::export]]
// arma::cube TestUpdateNu1(){
//   // Set space of functions
//   arma::vec t_obs =  arma::regspace(0, 10, 990);
//   splines2::BSpline bspline;
//   // Create Bspline object with 8 degrees of freedom
//   // 8 - 3 - 1 internal nodes
//   bspline = splines2::BSpline(t_obs, 8);
//   // Get Basis matrix (20 x 8)
//   arma::mat bspline_mat{bspline.basis(true)};
//   // Make B_obs
//   arma::field<arma::mat> B_obs(20,1);
//
//   for(int i = 0; i < 20; i++)
//   {
//     B_obs(i,0) = bspline_mat;
//   }
//
//   // Make nu matrix
//   arma::mat nu(3,8);
//   nu = {{2, 0, 1, 0, 0, 0, 1, 3},
//   {1, 3, 0, 2, 0, 0, 3, 0},
//   {5, 2, 5, 0, 3, 4, 1, 0}};
//
//
//   // Make Phi matrix
//   arma::cube Phi(3,8,5);
//   for(int i=0; i < 5; i++)
//   {
//     Phi.slice(i) = (5-i) * 0.1 * arma::randu<arma::mat>(3,8);
//   }
//   double sigma_sq = 0.01;
//
//   // Make chi matrix
//   arma::mat chi(20, 5, arma::fill::randn);
//
//
//   //Make Z
//   arma::mat Z(20, 3);
//   arma::vec c(3, arma::fill::ones);
//   arma::vec pi = BayesFPMM::rdirichlet(c);
//
//   // setting alpha_3 = 10
//   arma:: vec alpha = pi * 10;
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha).t();
//   }
//
//   arma::field<arma::vec> y_obs(20, 1);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 20; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 3; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs(j, 0) = arma::mvnrnd(B_obs(j, 0) * mean, sigma_sq *
//       arma::eye(B_obs(j,0).n_rows, B_obs(j,0).n_rows));
//   }
//
//   arma::cube Nu_samp = arma::randn(nu.n_rows, nu.n_cols, 500);
//   arma::vec b_1(nu.n_cols, arma::fill::zeros);
//   arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   arma::mat P(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   P.zeros();
//   for(int j = 0; j < P.n_rows; j++){
//     P(0,0) = 1;
//     if(j > 0){
//       P(j,j) = 2;
//       P(j-1,j) = -1;
//       P(j,j-1) = -1;
//     }
//     P(P.n_rows - 1, P.n_rows - 1) = 1;
//   }
//   arma::vec tau(nu.n_rows, arma::fill::ones);
//   tau = tau / 10;
//   for(int i = 0; i < 500; i++){
//     BayesFPMM::updateNu(y_obs, B_obs, tau, Phi, Z, chi, sigma_sq, i, 500,
//                         P, b_1, B_1, Nu_samp);
//   }
//   Nu_samp.save("/Users/nicholasmarco/Projects/BayesFPMM/inst/test-data/nu.txt", arma::arma_ascii);
//   arma::cube mod = arma::zeros(3,8,2);
//   arma::vec nu_ph = arma::zeros(200);
//   arma::mat nu_est = arma::zeros(3,8);
//   for(int i = 0; i < 3; i++){
//     for(int j = 0; j < 8; j++){
//       for(int k = 300; k < 500; k++){
//         nu_ph(k - 300) = Nu_samp(i,j,k);
//       }
//       nu_est(i,j) = arma::median(nu_ph);
//     }
//   }
//   mod.slice(1) = nu;
//   mod.slice(0) = nu_est;
//   return mod;
// }


// //' @export
// // [[Rcpp::export]]
// Rcpp::List Stuff(){
//   // Make nu matrix
//   arma::mat nu = arma::randn(2,8) * 2;
//
//
//   // Make Phi matrix
//   arma::cube Phi(2,8,2);
//   for(int i=0; i < 2; i++)
//   {
//     Phi.slice(i) = (2-i) * 0.2 * arma::randu<arma::mat>(2,8);
//   }
//   double sigma_sq = 0.001;
//
//   // Make chi matrix
//   arma::mat chi(100, 2, arma::fill::randn);
//
//
//   // Make Z matrix
//   arma::mat Z(100, 2);
//   arma::mat alpha(100,2, arma::fill::ones);
//   alpha = alpha * 10;
//   for(int i = 0; i < 20; i++){
//     alpha(i,0) = 10;
//     alpha(i,1) = 1;
//   }
//   for(int i = 20; i < 40; i++){
//     alpha(i,1) = 10;
//     alpha(i,0) = 1;
//   }
//   for(int i = 0; i < Z.n_rows; i++){
//     Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
//   }
//
//   arma::mat y_obs = arma::zeros(100, 8);
//   arma::vec mean = arma::zeros(8);
//
//   for(int j = 0; j < 100; j++){
//     mean = arma::zeros(8);
//     for(int l = 0; l < 2; l++){
//       mean = mean + Z(j,l) * nu.row(l).t();
//       for(int m = 0; m < Phi.n_slices; m++){
//         mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//       }
//     }
//     y_obs.row(j) = arma::mvnrnd(mean, sigma_sq *
//       arma::eye(mean.n_elem, mean.n_elem)).t();
//   }
//   Rcpp::List BestChain =  Rcpp::List::create(Rcpp::Named("nu", nu),
//                                              Rcpp::Named("chi", chi),
//                                              Rcpp::Named("sigma_sq",sigma_sq),
//                                              Rcpp::Named("Phi", Phi),
//                                              Rcpp::Named("Z", Z),
//                                              Rcpp::Named("Y",y_obs));
//   return(BestChain);
// }
//   //
//   //   arma::cube Nu_samp = arma::randn(nu.n_rows, nu.n_cols, 500);
//   //   arma::vec b_1(nu.n_cols, arma::fill::zeros);
//   //   arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   //   arma::mat P(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   //   P.zeros();
//   //   for(int j = 0; j < P.n_rows; j++){
//   //     P(0,0) = 1;
//   //     if(j > 0){
//   //       P(j,j) = 2;
//   //       P(j-1,j) = -1;
//   //       P(j,j-1) = -1;
//   //     }
//   //     P(P.n_rows - 1, P.n_rows - 1) = 1;
//   //   }
//   //   arma::vec tau(nu.n_rows, arma::fill::ones);
//   //   tau = tau / 10;
//   //   for(int i = 0; i < 500; i++){
//   //     BayesFPMM::updateNu(y_obs, B_obs, tau, Phi, Z, chi, sigma_sq, i, 500,
//   //                         P, b_1, B_1, Nu_samp);
//   //   }
//   //   Nu_samp.save("/Users/nicholasmarco/Projects/BayesFPMM/inst/test-data/nu.txt", arma::arma_ascii);
//   //   arma::cube mod = arma::zeros(3,8,2);
//   //   arma::vec nu_ph = arma::zeros(200);
//   //   arma::mat nu_est = arma::zeros(3,8);
//   //   for(int i = 0; i < 3; i++){
//   //     for(int j = 0; j < 8; j++){
//   //       for(int k = 300; k < 500; k++){
//   //         nu_ph(k - 300) = Nu_samp(i,j,k);
//   //       }
//   //       nu_est(i,j) = arma::median(nu_ph);
//   //     }
//   //   }
//   //   mod.slice(1) = nu;
//   //   mod.slice(0) = nu_est;
//   //   return mod;
// }
//
//
// //' @export
// // [[Rcpp::export]]
// Rcpp::List Stuff1(){
//   // arma::vec t_obs =  arma::regspace(0, 10, 990);
//   // splines2::BSpline bspline;
//   // // Create Bspline object with 8 degrees of freedom
//   // // 8 - 3 - 1 internal nodes
//   // bspline = splines2::BSpline(t_obs, 8);
//   // // Get Basis matrix (20 x 8)
//   // arma::mat bspline_mat{bspline.basis(true)};
//   // // Make B_obs
//   // arma::field<arma::mat> B_obs(20,1);
//   //
//   // for(int i = 0; i < 20; i++)
//   // {
//   //   B_obs(i,0) = bspline_mat;
//   // }
// int n_funct = 100;
//
// arma::field<arma::mat> t_obs1(n_funct,1);
// arma::mat t_obs = arma::zeros(12,2);
// t_obs.col(0) =  arma::regspace(0, 90, 990);
// t_obs.col(1) =  arma::regspace(0, 90, 990);
// t_obs1(0,0) = arma::zeros(144,2);
// int counter = 0;
// for(int i = 0; i < 12; i++){
//   for(int j = 0; j < 12; j++){
//     t_obs1(0,0)(counter, 0) = t_obs(i,0);
//     t_obs1(0,0)(counter, 1) = t_obs(j,0);
//     counter++;
//   }
// }
// for(int i = 1; i < n_funct; i++){
//   t_obs1(i,0) = t_obs1(0,0);
// }
//
// arma::field<arma::vec> internal_knots(2,1);
// internal_knots(0,0) = {250, 500, 750};
// internal_knots(1,0) = {250, 500, 750};
// arma::mat boundary_knots = {{0,990}, {0,990}};
//
// arma::vec basis_degree = {2,2};
//
// arma::field<arma::mat> B = BayesFPMM::TensorBSpline(t_obs1, n_funct, basis_degree,
//                                                     boundary_knots, internal_knots);
//
// // Make nu matrix
// arma::mat nu = arma::randn(2, 36) * 2;
//
// // Make Phi matrix
// arma::cube Phi(2,36,2);
// for(int i=0; i < 2; i++)
// {
//   Phi.slice(i) = (2-i) * 0.1 * arma::randu<arma::mat>(2,36);
// }
// double sigma_sq = 0.001;
//
// // Make chi matrix
// arma::mat chi(n_funct, 2, arma::fill::randn);
//
//
// // Make Z matrix
// arma::mat Z(n_funct, 2);
// arma::mat alpha(n_funct, 2, arma::fill::ones);
// alpha = alpha * 10;
// for(int i = 0; i < 20; i++){
//   alpha(i,0) = 10;
//   alpha(i,1) = 1;
// }
// for(int i = 20; i < 40; i++){
//   alpha(i,1) = 10;
//   alpha(i,0) = 1;
// }
// for(int i = 0; i < Z.n_rows; i++){
//   Z.row(i) = BayesFPMM::rdirichlet(alpha.row(i).t()).t();
// }
//
// arma::field<arma::vec> y_obs(n_funct, 1);
// arma::vec mean = arma::zeros(36);
// for(int j = 0; j < n_funct; j++){
//   mean = arma::zeros(36);
//   for(int l = 0; l < nu.n_rows; l++){
//     mean = mean + Z(j,l) * nu.row(l).t();
//     for(int m = 0; m < Phi.n_slices; m++){
//       mean = mean + Z(j,l) * chi(j,m) * Phi.slice(m).row(l).t();
//     }
//   }
//   y_obs(j,0) = arma::mvnrnd(B(j, 0) * mean, sigma_sq *
//     arma::eye(B(j,0).n_rows, B(j,0).n_rows));
// }
//
//   Rcpp::List BestChain =  Rcpp::List::create(Rcpp::Named("B", B),
//                                              Rcpp::Named("nu", nu),
//                                              Rcpp::Named("chi", chi),
//                                              Rcpp::Named("sigma_sq", sigma_sq),
//                                              Rcpp::Named("Phi", Phi),
//                                              Rcpp::Named("time", t_obs1),
//                                              Rcpp::Named("Z", Z),
//                                              Rcpp::Named("Y", y_obs));
//   return(BestChain);
// }
//   //
//   //   arma::cube Nu_samp = arma::randn(nu.n_rows, nu.n_cols, 500);
//   //   arma::vec b_1(nu.n_cols, arma::fill::zeros);
//   //   arma::mat B_1(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   //   arma::mat P(nu.n_cols, nu.n_cols, arma::fill::zeros);
//   //   P.zeros();
//   //   for(int j = 0; j < P.n_rows; j++){
//   //     P(0,0) = 1;
//   //     if(j > 0){
//   //       P(j,j) = 2;
//   //       P(j-1,j) = -1;
//   //       P(j,j-1) = -1;
//   //     }
//   //     P(P.n_rows - 1, P.n_rows - 1) = 1;
//   //   }
//   //   arma::vec tau(nu.n_rows, arma::fill::ones);
//   //   tau = tau / 10;
//   //   for(int i = 0; i < 500; i++){
//   //     BayesFPMM::updateNu(y_obs, B_obs, tau, Phi, Z, chi, sigma_sq, i, 500,
//   //                         P, b_1, B_1, Nu_samp);
//   //   }
//   //   Nu_samp.save("/Users/nicholasmarco/Projects/BayesFPMM/inst/test-data/nu.txt", arma::arma_ascii);
//   //   arma::cube mod = arma::zeros(3,8,2);
//   //   arma::vec nu_ph = arma::zeros(200);
//   //   arma::mat nu_est = arma::zeros(3,8);
//   //   for(int i = 0; i < 3; i++){
//   //     for(int j = 0; j < 8; j++){
//   //       for(int k = 300; k < 500; k++){
//   //         nu_ph(k - 300) = Nu_samp(i,j,k);
//   //       }
//   //       nu_est(i,j) = arma::median(nu_ph);
//   //     }
//   //   }
//   //   mod.slice(1) = nu;
//   //   mod.slice(0) = nu_est;
//   //   return mod;
// }
