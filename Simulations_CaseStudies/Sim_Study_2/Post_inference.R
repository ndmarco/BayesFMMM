library(BayesFMMM)
library(ggplot2)
library(gridExtra)
library(latex2exp)
library(pracma)

#################################################
## Change relevant directories  before running ##
#################################################
setwd()


BIC <- matrix(0, 10, 4)
AIC <- matrix(0, 10, 4)
DIC <- matrix(0, 10, 4)
avg_llik <- matrix(0, 10, 4)

time <- seq(0, 990, 10)
time <- rep(list(time), 200)

for(j in 1:4){
  if(j == 1){
    dir <- "./2_clusters/"
  }
  if(j == 2){
    dir <- "./3_clusters/"
  }
  if(j == 3){
    dir <- "./4_clusters/"
  }
  if(j == 4){
    dir <- "./5_clusters/"
  }
  for(i in 1:10){
    n_files <- 10
    basis_degree <- 3
    boundary_knots <- c(0, 1000)
    internal_knots <- c(200, 400, 600, 800)
    dir_i <- paste(dir, "trace", i, "/", sep = "")
    Y <- readRDS(paste("./data/data", i, ".RDS", sep = ""))
    Y <- Y$y
    AIC[i,j] <- Model_AIC(dir_i, 10, 1000, basis_degree, boundary_knots,
                          internal_knots, time, Y)
    BIC[i,j] <- Model_BIC(dir_i, 10, 1000, basis_degree, boundary_knots,
                          internal_knots, time, Y)
    DIC[i,j] <- Model_DIC(dir_i, 10, 1000, basis_degree, boundary_knots,
                          internal_knots, time, Y)
    x <- Model_LLik(dir_i, 10, 1000, basis_degree, boundary_knots, internal_knots, time, Y)
    avg_llik[i, j] <- mean(x[2000:10000])
    print(i)
    print(j)
  }
}

K_opt_output <- list(avg_llik, DIC, AIC, BIC)

loglik <- matrix(0, 40, 3)
loglik[1:10,1] <- (avg_llik[,1])
loglik[1:10,2] <- 2
loglik[1:10,3] <- seq(1,10, 1)
loglik[11:20,1] <- (avg_llik[,2])
loglik[11:20,2] <- 3
loglik[11:20,3] <- seq(1,10, 1)
loglik[21:30,1] <- (avg_llik[,3])
loglik[21:30,2] <- 4
loglik[21:30,3] <- seq(1,10, 1)
loglik[31:40,1] <- (avg_llik[,4])
loglik[31:40,2] <- 5
loglik[31:40,3] <- seq(1,10, 1)


loglik<- as.data.frame(loglik)
colnames(loglik) <- c("Log-Likelihood", "K", "sim")
loglik$K <- as.factor(loglik$K)
p4 <- ggplot(loglik, aes(x=K, y=`Log-Likelihood`, group = sim, col = sim)) +
  geom_point() +  geom_line() + theme_classic() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
                                                        legend.position = "none",plot.title = element_text(hjust = 0.5))

AIC_df <- matrix(0, 40, 3)
AIC_df[1:10,1] <- (AIC[,1])
AIC_df[1:10,2] <- 2
AIC_df[1:10,3] <- seq(1,10, 1)
AIC_df[11:20,1] <- (AIC[,2])
AIC_df[11:20,2] <- 3
AIC_df[11:20,3] <- seq(1,10, 1)
AIC_df[21:30,1] <- (AIC[,3])
AIC_df[21:30,2] <- 4
AIC_df[21:30,3] <- seq(1,10, 1)
AIC_df[31:40,1] <- (AIC[,4])
AIC_df[31:40,2] <- 5
AIC_df[31:40,3] <- seq(1,10, 1)


AIC_df<- as.data.frame(AIC_df)
colnames(AIC_df) <- c("AIC", "K", "sim")
AIC_df$K <- as.factor(AIC_df$K)
p1 <- ggplot(AIC_df, aes(x=K, y=AIC, group = sim, col = sim)) +
  geom_point() +  geom_line() + theme_classic() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
                                                        legend.position = "none",plot.title = element_text(hjust = 0.5))

BIC_df <- matrix(0, 40, 3)
BIC_df[1:10,1] <- (BIC[,1])
BIC_df[1:10,2] <- 2
BIC_df[1:10,3] <- seq(1,10, 1)
BIC_df[11:20,1] <- (BIC[,2])
BIC_df[11:20,2] <- 3
BIC_df[11:20,3] <- seq(1,10, 1)
BIC_df[21:30,1] <- (BIC[,3])
BIC_df[21:30,2] <- 4
BIC_df[21:30,3] <- seq(1,10, 1)
BIC_df[31:40,1] <- (BIC[,4])
BIC_df[31:40,2] <- 5
BIC_df[31:40,3] <- seq(1,10, 1)


BIC_df<- as.data.frame(BIC_df)
colnames(BIC_df) <- c("BIC", "K", "sim")
BIC_df$K <- as.factor(BIC_df$K)
p2 <- ggplot(BIC_df, aes(x=K, y=BIC, group = sim, col = sim)) +
  geom_point() +  geom_line() + theme_classic() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
                                                        legend.position = "none",plot.title = element_text(hjust = 0.5))

DIC_df <- matrix(0, 40, 3)
DIC_df[1:10,1] <- (DIC[,1])
DIC_df[1:10,2] <- 2
DIC_df[1:10,3] <- seq(1,10, 1)
DIC_df[11:20,1] <- (DIC[,2])
DIC_df[11:20,2] <- 3
DIC_df[11:20,3] <- seq(1,10, 1)
DIC_df[21:30,1] <- (DIC[,3])
DIC_df[21:30,2] <- 4
DIC_df[21:30,3] <- seq(1,10, 1)
DIC_df[31:40,1] <- (DIC[,4])
DIC_df[31:40,2] <- 5
DIC_df[31:40,3] <- seq(1,10, 1)


DIC_df<- as.data.frame(DIC_df)
colnames(DIC_df) <- c("DIC", "K", "sim")
DIC_df$K <- as.factor(DIC_df$K)
p3 <- ggplot(DIC_df, aes(x=K, y=DIC, group = sim, col = sim)) +
  geom_point() +  geom_line() + theme_classic() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
                                                        legend.position = "none",plot.title = element_text(hjust = 0.5))

grid.arrange(p1,p2,p3,p4,ncol =2)
