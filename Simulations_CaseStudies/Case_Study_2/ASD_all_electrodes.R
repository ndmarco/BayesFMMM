library(BayesFMMM)
library(eegkit)

#################################################################
## Change relevant directories and make folders before running ##
#################################################################

subj_id <- sort(c(10,	11,	13,	14,	15,	23,	26,	30,	31,	35,	48,	49,	50,
                  53,	54,	55,	161,165,	184,	188,	189,	195,	201,
                  # 202,	excluded due to low counts
                  207,	210,	213,	214,	242,	255,	261,	282,	283,
                  284,	286,	287,	289,	290,	343,	351,	2,	3,	5,	6,
                  7,	8,	9,	12,	18,	19,	22,	24,	25,	27,	33,	34,	37,	38,
                  40,	41,	42,	43,	44,	47,	51,	401,	405,	406,	408,	411,
                  415,	416,	417,	418,	423,	426,	427,	430,
                  #431,	excluded due to low counts
                  433,	436,	438,	439,	440,	442,	444,	445,	446,	447,
                  448,	450,	451,	452,	453,	3019,	3024,	3026,	3029,	3032))

load("/Users/user/Projects/Simulation/real_data/pa.dat.Rdata")
chan_id <- c('FP1', 'FP2','F9','F7','F3','FZ','F4','F8','F10','T9','T7',
             'C3','CZ','C4','T8','T10','P9','P7','P3','PZ','P4','P8','P10','O1','O2')
## Get coordinates
data("eegcoord")
electrode_loc <- eegcoord[match(chan_id,rownames(eegcoord)),]

## Create list of observed points
Y <- list(pa.dat$y[pa.dat$ID == subj_id[1]])
for(i in 2:length(subj_id)){
  Y_i <- list(pa.dat$y[pa.dat$ID == subj_id[i]])
  Y <- append(Y, Y_i)
}

time <- matrix(0, nrow = 825, ncol = 3)
for(i in 1:825){
  time[i,1] <- electrode_loc$xproj[pa.dat$reg[pa.dat$ID == subj_id[1]][i]]
  time[i,2] <- electrode_loc$yproj[pa.dat$reg[pa.dat$ID == subj_id[1]][i]]
  time[i,3] <- pa.dat$func[pa.dat$ID == subj_id[1]][i]
}
time <- list(time)


for(j in 2:97){
  time_j <- matrix(0, nrow = 825, ncol = 3)
  for(i in 1:825){
    time_j[i,1] <- electrode_loc$xproj[pa.dat$reg[pa.dat$ID == subj_id[j]][i]]
    time_j[i,2] <- electrode_loc$yproj[pa.dat$reg[pa.dat$ID == subj_id[j]][i]]
    time_j[i,3] <- pa.dat$func[pa.dat$ID == subj_id[j]][i]
  }
  time_j <- list(time_j)
  time <- append(time, time_j)
}

### set hyperparameters
tot_mcmc_iters <- 10000
n_try <- 100
k <- 2
n_funct <- 97
basis_degree <- c(2,2,2)
n_eigen <- 2
boundary_knots <- matrix(c(-14, 14, -12, 12.5, 6, 14), nrow = 3, byrow = T)
internal_knots1 <- c(-7, 0, 7)
internal_knots2 <- c(-6, 0, 6)
internal_knots3 <- c(8.33, 11.66)
internal_knots <- list(internal_knots1, internal_knots2, internal_knots3)

## Run function
est1 <- BHDFMMM_Nu_Z_multiple_try(tot_mcmc_iters, n_try, k, Y, time, n_funct,
                                  basis_degree, n_eigen, boundary_knots,
                                  internal_knots)

## Run function
est2 <- BHDFMMM_Theta_est(tot_mcmc_iters, k, Y, time, n_funct,
                          basis_degree, n_eigen, boundary_knots,
                          internal_knots, est1$Z, est1$nu)

MCMC.chain <-BHDFMMM_warm_start(tot_mcmc_iters, k, Y, time, n_funct,
                                basis_degree, n_eigen, boundary_knots,
                                internal_knots, est1$Z, est1$pi, est1$alpha_3,
                                est2$delta, est2$gamma, est2$Phi, est2$A,
                                est1$nu, est1$tau, est2$sigma, est2$chi)


