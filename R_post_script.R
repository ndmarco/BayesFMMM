library(BayesFPMM)

######
######
######
## Run SImulations
##
library(BayesFPMM)
for(i in 1:20){
  data_dir <- "/Users/nicholasmarco/Projects/Simulation/integrated_error/25_obs/data/"
  x <- TestBFPMM_Nu_Z_multiple_try(2000, 0.001, 1/2, 10, 10000, 500, 2, data_dir)
  y <- TestBFPMM_Theta(5000, 0.001, x$Z, x$nu, 0.8, 2, data_dir)
  dir <- paste("/Users/nicholasmarco/Projects/Simulation/integrated_error/25_obs/trace", as.character(i), "/", sep = "")
  #dir <- "C:\\Projects\\Simulation\\Optimal_K\\2_clusters\\"
  z <- TestBFPMM_MTT_warm_start(1/5, 10, 1000000, 500000, 10000, dir, 0.001, y$Z,
                                x$pi, x$alpha_3, y$delta, y$gamma, y$Phi, y$A, y$nu,
                                x$tau, y$sigma, y$chi, 0.8, 2, data_dir)
  saveRDS(z, paste(dir, "x_results.RDS", sep = ""))
}

x <- readRDS("c:\\Projects\\Simulation\\High_Variance\\Trace1\\x_results.RDS")
nu_true <- x$nu_true
all_nu <- array(0, dim = c(2,8,3500,10))
all_Z <- array(0, dim =c(100,2,3500,10))
all_f1_97_5 <- array(0,dim = c(100, 10))
all_f2_2_5 <- array(0,dim = c(100,10))
all_f2_97_5 <- array(0,dim = c(100, 10))
all_f1_2_5 <- array(0,dim = c(100,10))
all_Z_97_5 <- array(0,dim = c(100, 10))
all_Z_2_5 <- array(0,dim = c(100,10))
for(j in 1:10){
  print(j)
  nu <- array(0,dim=c(2,8,4000))
  dir = paste("c:\\Projects\\Simulation\\High_Variance\\Trace",j,"\\Nu", sep ="")
  for(i in 0:19){
    nu_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
    nu[,,(200*(i) + 1):(200*(i+1))] <- nu_i
  }
  Z <- array(0,dim=c(100,2,10000))
  dir = "/Users/nicholasmarco/Projects/Simulation/2_cluster/trace4/Z"
  for(i in 0:49){
    Z_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
    Z[,,(200*(i) + 1):(200*(i+1))] <- Z_i
  }
  nu <- nu[,,501:4000]
  Z <- Z[,,501:10000]

  ## Fix label switching problem
  nu_ph <- nu
  Z_ph <- Z
  nu_ph[1,,] <- nu[2,,]
  Z_ph[,1,] <- Z[,2,]
  nu_ph[2,,] <- nu[1,,]
  Z_ph[,2,] <- Z[,1,]
  if(sum(abs(nu[,,3500]-nu_true)) > sum(abs(nu_ph[,,3500] - nu_true))){
    all_nu[,,,j] <- nu_ph
    all_Z[,,,j] <- Z_ph
    nu <- nu_ph
    Z <- Z_ph
  }else{
    all_nu[,,,j] <- nu
    all_Z[,,,i=j] <- Z
  }

  y <- GetStuff(0.001)
  x <- readRDS("c:\\Projects\\Simulation\\High_Variance\\Trace2\\x_results.RDS")
  f_obs1 <- matrix(0, 2500, 100)
  for(i in 1:1500){
    f_obs1[i,] <- t(y$B[[1]] %*% t(t(nu[1,,i])))
  }

  f1_97_5 <- rep(0,100)
  f1_2_5 <- rep(0,100)
  for(i in 1:100){
    q <- quantile(f_obs1[,i], probs = c(0.025, 0.975))
    f1_2_5[i] <- q[1]
    f1_97_5[i] <- q[2]
  }
  all_f1_2_5[,j]<- f1_2_5
  all_f1_97_5[,j] <- f1_97_5

  f_obs2 <- matrix(0, 3500, 100)
  for(i in 1:3500){
    f_obs2[i,] <- t(y$B[[1]] %*% t(t(nu[2,,i])))
  }

  f2_97_5 <- rep(0,100)
  f2_2_5 <- rep(0,100)
  for(i in 1:100){
    q <- quantile(f_obs2[,i], probs = c(0.025, 0.975))
    f2_2_5[i] <- q[1]
    f2_97_5[i] <- q[2]
  }
  all_f2_2_5[,j]<- f2_2_5
  all_f2_97_5[,j] <- f2_97_5


  Z_97_5 <- rep(0,100)
  Z_2_5 <- rep(0,100)
  for(i in 1:100){
    q <- quantile(Z[i,1,], probs = c(0.025, 0.975))
    Z_2_5[i] <- q[1]
    Z_97_5[i] <- q[2]
  }

  all_Z_2_5[,j] <- Z_2_5
  all_Z_97_5[,j] <- Z_97_5

}

f1_true <- rep(0,100)
f1_true <- x$nu_true[2,] %*% t(y$B[[1]])
plot(f1_true[1,], type = "l", ylim = c(-3,4), lwd = 3)
for(i in 1:10){
  lines(all_f2_2_5[,i], col=i, lty = 2)
  lines(all_f2_97_5[,i], col=i, lty=2)
}

library(plotrix)
Z_true <- x$Z_true[1:10,1]
Z_true <- rep(Z_true, 10)
Z_true <- (as.vector(all_Z_97_5[1:10,1:10]) + as.vector(all_Z_2_5[1:10,1:10]))/2
obs_num <- 1:10
obs_num <- rep(obs_num, 10)
col_num <- 1:100
for(i in 1:10){
  for(j in 1:10){
    col_num[(i-1)*10 + j] <- rainbow(10)[i]
  }
}

plot(1:10, x$Z_true[1:10,1])
plotCI(obs_num, Z_true, ui=as.vector(all_Z_97_5[1:10,1:10]), li = as.vector(all_Z_2_5[1:10,1:10]), scol = col_num, col ="#1C00ff00", add=T)

######
######
######
######
## Get Z estimates
Z <- array(0,dim=c(100,3,4000))
dir = "C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\Z"
for(i in 0:19){
  Z_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
  Z[,,(200*(i) + 1):(200*(i+1))] <- Z_i
}


## Get nu estimates
nu <- array(0,dim=c(3,8,4000))
dir = "C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\Nu"
for(i in 0:19){
  nu_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
  nu[,,(200*(i) + 1):(200*(i+1))] <- nu_i
}

chi <- array(0, dim=c(100,3,4000))
dir = "C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\Chi"
for(i in 0:19){
  chi_i <- TestReadCube(paste(dir, as.character(i),".txt", sep = ""))
  chi[,,(200*(i) + 1):(200*(i+1))] <- chi_i
}

Phi <- array(0,dim=c(3,8,3,4000))
dir = "C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\Phi"
for(i in 0:19){
  Phi_i <- TestReadField(paste(dir, as.character(i),".txt", sep = ""))
  for(j in 1:200){
    Phi[,,,(i)*200 + j] = Phi_i[[j]]
  }
}

##
## Calculate DIC and BIC
##

## Estimate curve level fit
##
##
##
Phi <- Phi[,,,500:4000]
nu <- nu[,,500:4000]
Z <- Z[,,500:4000]
chi <- chi[,,500:4000]

y <- GetStuff(0.001)
x <- readRDS("c:\\Projects\\trace\\x_results.RDS")

for(k in 1:100){
  f_i <- matrix(0, 3500, 100)
  fun = k
  for(i in 1:3500){
    f_i[i,] <- f_i[i,] + Z[fun, , i] %*% (nu[,,i]) %*% t(y$B[[1]])
    for(j in 1:3){
      f_i[i,] <- f_i[i,] + chi[fun,j,i] * Z[fun,,i] %*% Phi[,,j, i] %*% t(y$B[[1]])
    }
  }

  for(i in 1:100){
    z <- quantile(f_i[,i], probs = c(0.025, 0.975))
    f3_2_5[i] <- z[1]
    f3_97_5[i] <- z[2]
  }
  plot(x$y_obs[[fun]], type = 'l', ylab = "observed data")
  lines(f3_2_5, col = "red")
  lines(f3_97_5, col = "red")
}

##
##
##

## Get rid of burn-in
nu <- nu[,,1001:4000]
y <- GetStuff(0.001)
x <- readRDS("C:\\Projects\\Simulation\\Optimal_K\\3_clusters\\x_results.RDS")
f_obs1 <- matrix(0, 1500, 100)
for(i in 1:1500){
  f_obs1[i,] <- t(y$B[[1]] %*% t(t(nu[1,,i])))
}
f1_97_5 <- rep(0,100)
f1_2_5 <- rep(0,100)
for(i in 1:100){
  z <- quantile(f_obs1[,i], probs = c(0.025, 0.975))
  f1_2_5[i] <- z[1]
  f1_97_5[i] <- z[2]
}
f1_true <- rep(0,100)
f1_true <- x$nu_true[3,] %*% t(y$B[[1]])


plot(f1_true[1,], type = 'l', ylab = "function 1")
lines(f1_2_5, col = "red")
lines(f1_97_5, col = "red")

## Function 2
f_obs2 <- matrix(0, 1000, 100)
for(i in 1:1000){
  f_obs2[i,] <- t(y$B[[1]] %*% t(t(nu[2,,i])))
}
f2_97_5 <- rep(0,100)
f2_2_5 <- rep(0,100)
for(i in 1:100){
  z <- quantile(f_obs2[,i], probs = c(0.025, 0.975))
  f2_2_5[i] <- z[1]
  f2_97_5[i] <- z[2]
}
f2_true <- rep(0,100)
f2_true <- x$nu_true[2,] %*% t(y$B[[1]])

plot(f2_true[1,], type = 'l', ylab = "function 2")
lines(f2_2_5, col = "red")
lines(f2_97_5, col = "red")

## Function 3

f_obs3 <- matrix(0, 1500, 100)
for(i in 1:1500){
  f_obs3[i,] <- t(y$B[[1]] %*% t(t(nu[3,,i])))
}
f3_97_5 <- rep(0,100)
f3_2_5 <- rep(0,100)
for(i in 1:100){
  z <- quantile(f_obs3[,i], probs = c(0.025, 0.975))
  f3_2_5[i] <- z[1]
  f3_97_5[i] <- z[2]
}
f3_true <- rep(0,100)
f3_true <- x$nu_true[1,] %*% t(y$B[[1]])

plot(f3_true[1,], type = 'l', ylab = "function 3")
lines(f3_2_5, col = "red")
lines(f3_97_5, col = "red")


### Phi

## Get nu estimates
Phi <- array(0,dim=c(2,8,3,600))
dir = "c:\\Projects\\trace\\Phi"
for(i in 0:2){
  Phi_i <- TestReadField(paste(dir, as.character(i),".txt", sep = ""))
  for(j in 1:200){
    Phi[,,,(i)*200 + j] = Phi_i[[j]]
  }
}

## Get rid of Burn-in
Phi <- Phi[,,,200:600]


f1_var <- array(0, dim= c(100, 100, 400))
for(i in 1:400){
  for(j in 1:100){
    for(k in 1:100){
      for(m in 1:3){
        f1_var[j,k,i] <- f1_var[j,k,i] + Phi[1, ,m ,i] %*% t(t(y$B[[1]][j,])) %*% y$B[[1]][k,] %*% t(t(Phi[1, ,m ,i]))
      }
    }
  }
  print(i)
}
cov1_97_5 <- matrix(0,100,100)
cov1_2_5 <- matrix(0,100,100)
for(j in 1:100){
  for(k in 1:100){
    z <- quantile(f1_var[j,k,], probs = c(0.025, 0.975))
    cov1_2_5[j,k] <- z[1]
    cov1_97_5[j,k] <- z[2]
  }
}

cov1_true <- matrix(0, 100, 100)

y <- readRDS("c:\\Projects\\trace\\x_results.RDS")
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov1_true[j,k] <- cov1_true[j,k] + y$Phi_true[1, ,m] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(y$Phi_true[1, ,m]))
    }
  }
}

library(plotly)

fig <- plot_ly(showscale = FALSE)
fig <- fig %>% add_surface(z = ~cov1_true)
fig <- fig %>% add_surface(z = ~cov1_2_5, opacity = 0.50)
fig <- fig %>% add_surface(z = ~cov1_97_5, opacity = 0.50)

fig



#### Cov 2

f2_var <- array(0, dim= c(100, 100, 3000))
for(i in 1:3000){
  for(j in 1:100){
    for(k in 1:100){
      for(m in 1:3){
        f2_var[j,k,i] <- f2_var[j,k,i] + Phi[2, ,m ,i] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(Phi[2, ,m ,i]))
      }
    }
  }
  print(i)
}
cov2_97_5 <- matrix(0,100,100)
cov2_2_5 <- matrix(0,100,100)
for(j in 1:100){
  for(k in 1:100){
    z <- quantile(f2_var[j,k,], probs = c(0.025, 0.975))
    cov2_2_5[j,k] <- z[1]
    cov2_97_5[j,k] <- z[2]
  }
}

cov2_true <- matrix(0, 100, 100)

y <- readRDS("/Users/nicholasmarco/Projects/FDA/Trace/rstudio-export/x_results.RDS")
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov2_true[j,k] <- cov2_true[j,k] + y$Phi_true[2, ,m] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(y$Phi_true[2, ,m]))
    }
  }
}

library(plotly)

fig2 <- plot_ly(showscale = FALSE)
fig2 <- fig2 %>% add_surface(z = ~cov2_true)
fig2 <- fig2 %>% add_surface(z = ~cov2_2_5, opacity = 0.50)
fig2 <- fig2 %>% add_surface(z = ~cov2_97_5, opacity = 0.50)

fig2


### Cov 3

f3_var <- array(0, dim= c(100, 100, 3000))
for(i in 1:3000){
  for(j in 1:100){
    for(k in 1:100){
      for(m in 1:3){
        f3_var[j,k,i] <- f3_var[j,k,i] + Phi[3, ,m ,i] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(Phi[3, ,m ,i]))
      }
    }
  }
  print(i)
}
cov3_97_5 <- matrix(0,100,100)
cov3_2_5 <- matrix(0,100,100)
for(j in 1:100){
  for(k in 1:100){
    z <- quantile(f3_var[j,k,], probs = c(0.025, 0.975))
    cov3_2_5[j,k] <- z[1]
    cov3_97_5[j,k] <- z[2]
  }
}

cov3_true <- matrix(0, 100, 100)

y <- readRDS("/Users/nicholasmarco/Projects/FDA/Trace/rstudio-export/x_results.RDS")
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov3_true[j,k] <- cov3_true[j,k] + y$Phi_true[3, ,m] %*% t(t(x$B[[1]][j,])) %*% x$B[[1]][k,] %*% t(t(y$Phi_true[3, ,m]))
    }
  }
}

library(plotly)

fig3 <- plot_ly(showscale = FALSE)
fig3 <- fig3 %>% add_surface(z = ~cov3_true)
fig3 <- fig3 %>% add_surface(z = ~cov3_2_5, opacity = 0.50)
fig3 <- fig3 %>% add_surface(z = ~cov3_97_5, opacity = 0.50)

fig3

### Real Case study

setwd("/Users/nicholasmarco/Projects/Simulation/real_data")

### Peak alpha data for John
library(pracma)

# Subject ID
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
# Channel ID (order of chan_id corresponds to 1:25 labeling of regions)
chan_id <- c('Fp1', 'Fp2','F9','F7','F3','Fz','F4','F8','F10','T9','T7',
             'C3','Cz','C4','T8','T10','P9','P7','P3','Pz','P4','P8','P10','O1','O2')

# Demographic Data
demDat <- read.csv(file='demographic_data.csv', header = TRUE)
colnames(demDat) <- c("ID", "Gender", "Age", "Group", "VIQ", "NVIQ")
demDat <- demDat[which(demDat$ID %in% subj_id), ]

# Peak Alpha Data
load("pa.dat.Rdata")
# ID: subject ID
# group: TD(1) or ASD (2)
# func: frequency domain
# reg: electrode (order corresponds to chan_id above)
# Age: age in months
# y: alpha spectra density
out1 <- unique(pa.dat$func)
out3 <- unique(pa.dat$reg)
matplot(matrix(pa.dat$y, nrow = length(out1)), type = "l") # data
trapz(out1, pa.dat$y[1:33]) # all functional observations integrate to 1 (normalized across electordes, subjects)

### Convert to wide format
Y <- pa.dat
## paper used T8 electrode
Y <- Y[Y$reg == 15,]
Y$ID <- paste(Y$ID, Y$reg, sep = ".")
Y <- reshape(Y[,c(1,3,6)], idvar = "ID", timevar = "func", direction = "wide")
Y <- Y[,-1]
Y <- as.matrix(Y)

#get rid of ID value
library(reshape2)
library(ggplot2)
Y <- split(Y, seq(nrow(Y)))
time <- seq(6, 14, 0.25)
data_g1 <- Y[demDat$Group ==2 & demDat$Age > 100,]
colnames(data_g1) <-  time
data_g1 <- melt(data_g1)
data_g1$Var1 <- as.factor(data_g1$Var1)
p1 <- ggplot(data = data_g1, aes(x = Var2, y = value, colour = Var1)) + geom_line() + xlab("Frequency (Hz)") + ylab("Power") + ggtitle("ASD") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5),
        legend.position = "none")

data_g2 <- Y[demDat$Group ==1& demDat$Age > 100,]
colnames(data_g2) <-  time
data_g2 <- melt(data_g2)
data_g2$Var1 <- as.factor(data_g2$Var1)
p2 <- ggplot(data = data_g2, aes(x = Var2, y = value, colour = Var1)) + geom_line() + xlab("Frequency (Hz)") + ylab("Power") + ggtitle("TD") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5),
        legend.position = "none")
#matplot( t(time_mat[demDat$Group ==2 & demDat$Age > 100,]), t(Y[demDat$Group ==2 & demDat$Age > 100,]), type = 'l', xlab = "Frequency (Hz)", ylab = "Power", main = "ASD")
#matplot( t(time_mat[demDat$Group ==1 & demDat$Age > 100,]), t(Y[demDat$Group ==1 & demDat$Age > 100,]), type = 'l', xlab = "Frequency (Hz)", ylab = "Power", main = "TD")
grid.arrange(p1, p2, ncol = 2)

time <- seq(6, 14, 0.25)
time <- rep(list(time), 97)
time_mat <- matrix(rep(time, 97), nrow = 97, byrow = T)

#############################
### Real Data example #######
#############################

### Start MCMC

x <- BFPMM_Nu_Z_multiple_try(2000, 200, 2, Y, time, 97, 8, 3)
y <- BFPMM_Theta_Est(5000, x$Z, x$nu, 0.8, 2, Y, time, 97, 8, 3)
dir <- "/Users/nicholasmarco/Projects/Simulation/real_data/trace2/"
z <- BFPMM_warm_start(1/5, 10, 1000000, 500000, 10000, x$Z,
                      x$pi, x$alpha_3, y$delta, y$gamma, y$Phi, y$A, x$nu,
                      x$tau, y$sigma, y$chi, 0.8, 2, Y, time, 97, 8, 3, 50, dir)
saveRDS(z, paste(dir, "x_results.RDS", sep = ""))

### get credible intervals for mean
mean_1 <- GetMeanCI_S(dir,50, time[[1]], 1)
plot(time[[1]],mean_1$CI_50, type = 'l')
lines(time[[1]], mean_1$CI_025, col = "red")
lines(time[[1]], mean_1$CI_975, col = "red")

mean_2 <- GetMeanCI_S(dir,50, time[[1]], 2)
plot(time[[1]],mean_2$CI_50, type = 'l', xlab = "Frequency (Hz)", ylab = "Power", ylim = c(0, 0.4))
lines(time[[1]],mean_2$CI_025, col = "red")
lines(time[[1]],mean_2$CI_975, col = "red")


predframe <- data.frame(freq = time[[1]],
                        median=mean_1$CI_50,lwr=mean_1$CI_025,upr=mean_1$CI_975)
p1 <- ggplot(predframe, aes(freq, median))+
  geom_line(col = "blue")+
  geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3)  + ylab("Power") +
  xlab("Frequency (Hz)") + ylim(c(0, 0.4)) + xlim(c(6,14)) + ggtitle("Mean 1") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))

predframe <- data.frame(freq = time[[1]],
                        median=mean_2$CI_50,lwr=mean_2$CI_025,upr=mean_2$CI_975)
p2<- ggplot(predframe, aes(freq, median))+
  geom_line(col = "blue")+
  geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3) + ylab("Power")+
  xlab("Frequency (Hz)") + ylim(c(0, 0.4)) + xlim(c(6,14)) + ggtitle("Mean 2") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))

grid.arrange(p1, p2, ncol = 2)
x <- arrangeGrob(p1, p2)
##ggsave("/Users/nicholasmarco/Projects/BayesFPMM/Paper/means.jpeg", arrangeGrob(p1, p2))
##p1

### Get median cluster memberships

Z_post <- GetZCI(dir, 50)
data_Z <- data.frame("Cluster 1" = Z_post$CI_50[,1], "Clinical Diagnosis" = demDat$Group)
data_Z$Clinical.Diagnosis[data_Z$Clinical.Diagnosis == 2] <- "ASD"
data_Z$Clinical.Diagnosis[data_Z$Clinical.Diagnosis == 1] <- "TD"
ggplot(data= data_Z, aes(x = `Cluster.1` , y = Clinical.Diagnosis)) + geom_violin(trim = F, xlim = c(0,1)) + geom_point() + xlab("Cluster 1") + ylab("Clinical Diagnosis") +
  stat_summary(
    geom = "point",
    fun.x = "mean",
    col = "black",
    size = 3,
    shape = 24,
    fill = "red")+ xlim(c(0,1)) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                        panel.background = element_blank(),axis.line = element_line(colour = "black"),
                                        plot.title = element_text(hjust = 0.5))
data_VIQ <- data_Z <- data.frame("Cluster 1" = Z_post$CI_50[,1], "VIQ" = demDat$VIQ)
p1 <- ggplot(data= data_Z, aes(x = `Cluster.1` , y = VIQ)) + geom_point() + xlab("Cluster 1") + ylab("Verbal IQ") +
  geom_smooth(method='lm', colour = "red") + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                   panel.background = element_blank(),axis.line = element_line(colour = "black"),
                                                   plot.title = element_text(hjust = 0.5))
data_NVIQ <- data_Z <- data.frame("Cluster 1" = Z_post$CI_50[,1], "NVIQ" = demDat$NVIQ)
p2 <- ggplot(data= data_Z, aes(x = `Cluster.1` , y = NVIQ)) + geom_point() + xlab("Cluster 1") + ylab("Nonverbal IQ") +
  geom_smooth(method='lm', colour = "red") + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                   panel.background = element_blank(),axis.line = element_line(colour = "black"),
                                                   plot.title = element_text(hjust = 0.5))
grid.arrange(p1, p2, ncol = 2)

## Get Covariances
Cov_1 <- GetCovCI_S(dir, 50, 200, time[[1]], time[[1]], 1,1)

library(plotly)

fig <- plot_ly(showscale = FALSE)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_50)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_975, opacity = 0.20)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_025, opacity = 0.20)

fig


##################################
##### Simulation Study 1 #########
##################################

### Pointwise Credible Interval coverage mean
dir <- "/Users/nicholasmarco/Projects/Simulation/integrated_error/"

y <- GetStuff(0.0000001,"/Users/nicholasmarco/Projects/Simulation/integrated_error/100_obs/data/", 100)
x <- readRDS("/Users/nicholasmarco/Projects/Simulation/integrated_error/100_obs/trace1/x_results.RDS")
nu_1_true <-  y$B[[1]] %*% t(t(x$nu_true[1,]))
nu_2_true <-  y$B[[1]] %*% t(t(x$nu_true[2,]))

##plot functions



observed <- matrix(0, 3, 100)
observed[1,] <- y$y[[1]]
observed[2,] <- y$y[[40]]
observed[3,] <- y$y[[84]]
obs1 <- melt(observed)
obs1$Var2 <- (obs1$Var2 -1) * 10


observed1 <- data.frame("funct" =observed, "Z" =x$Z_true[c(1,40, 84),1], id = 1:3)

rbPal <- colorRampPalette(c('red','blue'))
observed1$Col <- rbPal(30)[as.numeric(cut(observed1$Z,breaks = 30))]
col2 <- observed1$Col[2]
observed1$Col[2] <- observed1$Col[3]
observed1$Col[3] <- col2
obs <- melt(observed1, id = c("id", "Z", "Col"))

obs1$Z <- obs$Z
obs1$Z <- as.factor(obs1$Z)
obs1$Col <- obs$Col
levels(obs1$Z) <- c("0", "0.5", "1")

ggplot(obs1, aes(x = Var2, y = value, colour=Z))+
  geom_line(aes(colour = Z))+ scale_colour_manual(values = observed1$Col)+
  xlab("") + ylab("") + labs(colour = "Cluster 1") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))

cov1_true <- matrix(0, 100, 100)
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov1_true[j,k] <- cov1_true[j,k] + x$Phi_true[1, ,m] %*% t(t(y$B[[1]][j,])) %*% y$B[[1]][k,] %*% t(t(x$Phi_true[1, ,m]))
    }
  }
}

cov2_true <- matrix(0, 100, 100)
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov2_true[j,k] <- cov2_true[j,k] + x$Phi_true[2, ,m] %*% t(t(y$B[[1]][j,])) %*% y$B[[1]][k,] %*% t(t(x$Phi_true[2, ,m]))
    }
  }
}

cov12_true <- matrix(0, 100, 100)
for(j in 1:100){
  for(k in 1:100){
    for(m in 1:3){
      cov12_true[j,k] <- cov12_true[j,k] + x$Phi_true[2, ,m] %*% t(t(y$B[[1]][j,])) %*% y$B[[1]][k,] %*% t(t(x$Phi_true[1, ,m]))
    }
  }
}

time <- seq(0, 990, 10)

int_err_mean1 <- matrix(0, 25, 3)
int_err_mean2 <- matrix(0, 25, 3)

int_err_cov1 <- matrix(0, 25, 3)
int_err_cov2 <- matrix(0, 25, 3)
int_err_cov12 <- matrix(0, 25, 3)

for(j in 1:3){
  if(j == 1){
    dir <- "/Users/nicholasmarco/Projects/Simulation/integrated_error/25_obs/"
  }
  if(j == 2){
    dir <- "/Users/nicholasmarco/Projects/Simulation/integrated_error/50_obs/"
  }
  if(j == 3){
    dir <- "/Users/nicholasmarco/Projects/Simulation/integrated_error/100_obs/"
  }
  for(i in 1:25){
    dir_i <- paste(dir, "trace", i, "/", sep = "")
    nu_1 <- GetMeanCI_PW(dir_i, 50, time, 1)
    nu_2 <- GetMeanCI_PW(dir_i, 50, time, 2)
    cov1 <- GetCovCI_S(dir_i, 50, 200, time, time, 0, 0)
    cov2 <- GetCovCI_S(dir_i, 50, 200, time, time, 1, 1)
    cov12 <- GetCovCI_S(dir_i, 50, 200, time, time, 0, 1)
    Z <- GetZCI(dir_i, 50)
    ## Label Switching
    if(sum(abs(nu_1$CI_50 - nu_1_true)) > sum(abs(nu_1$CI_50 - nu_2_true))){
      nu_i <- nu_1
      nu_1 <- nu_2
      nu_2 <- nu_i
      cov_i <- cov2
      cov1 <- cov2
      cov2 <- cov_i
      cov12$CI_50<- t(cov12$CI_50)
    }
    int_err_mean1[i,j] <- sum((nu_1$CI_50 - nu_1_true)^2) * 10
    int_err_mean2[i,j] <- sum((nu_2$CI_50 - nu_2_true)^2) * 10
    int_err_cov1[i,j] <- sum((cov1$CI_50 - cov1_true)^2) * 10^2
    int_err_cov2[i,j] <- sum((cov2$CI_50 - cov2_true)^2) * 10^2
    int_err_cov12[i,j] <- sum((cov12$CI_50 - cov12_true)^2) * 10^2
    print(i)
    print(j)
  }
}

rel_int_err_mean1 <- sum((nu_1$CI_50)^2) * 10
rel_int_err_mean2 <- sum((nu_2$CI_50)^2) * 10
rel_int_err_cov1 <- sum((cov1$CI_50)^2) * 100
rel_int_err_cov2 <- sum((cov2$CI_50)^2) * 100
rel_int_err_cov12 <- sum((cov12$CI_50)^2) * 100


dir <- "/Users/nicholasmarco/Projects/Simulation/integrated_error/100_obs/trace1/"

time <- seq(0, 990, 10)
Cov_1 <- GetCovCI_S(dir, 50, 200, time, time, 2,2)
library(plotly)

fig <- plot_ly(showscale = FALSE)
fig <- fig %>% add_surface(z = ~ cov2_true)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_975, opacity = 0.20)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_025, opacity = 0.20)

fig

time <- seq(0, 990, 10)
Cov_1 <- GetCovCI_S(dir, 50, 200, time, time, 1,1)
library(plotly)

fig <- plot_ly(showscale = FALSE)
fig <- fig %>% add_surface(z = ~ cov1_true)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_975, opacity = 0.20)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_025, opacity = 0.20)

fig

time <- seq(0, 990, 10)
Cov_1 <- GetCovCI_S(dir, 50, 200, time, time, 2,1)
library(plotly)

fig <- plot_ly(showscale = FALSE)
fig <- fig %>% add_surface(z = ~ cov12_true)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_975, opacity = 0.20)
fig <- fig %>% add_surface(z = ~ Cov_1$CI_025, opacity = 0.20)

fig

mean_1_pw <- GetMeanCI_PW(dir, 50, time, 1)
mean_1_S <- GetMeanCI_S(dir, 50, time, 1)
predframe_pw <- data.frame(freq = time,
                        true_func=nu_1_true,lwr=mean_1_pw$CI_025,upr=mean_1_pw$CI_975)
predframe_S <- data.frame(freq = time,
                           true_func=nu_1_true,lwr=mean_1_S$CI_025,upr=mean_1_S$CI_975)
p1 <- ggplot(predframe_pw, aes(freq, true_func))+
  geom_line(col = "blue")+
  geom_ribbon(data=predframe_pw,aes(ymin=lwr,ymax=upr),alpha=0.5)  + ylab("") +
  xlab("") + ggtitle("Pointwise CI") + ylim(c(-1,2)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))
p2 <- ggplot(predframe_S, aes(freq, true_func))+
  geom_line(col = "blue")+
  geom_ribbon(data=predframe_S,aes(ymin=lwr,ymax=upr),alpha=0.5)  + ylab("") +
  xlab("") + ggtitle("Simultaneous CI") + ylim(c(-1,2)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))

grid.arrange(p1, p2, ncol = 2)
### Simultaneous Credible Interval coverage mean
dir <- "/Users/nicholasmarco/Projects/Simulation/integrated_error/80_obs/"

counter1 <- 0
counter2 <- 0
counterall <- 0
y <- GetStuff(0.001,"/Users/nicholasmarco/Projects/Simulation/integrated_error/80_obs/data/", 80)
x <- readRDS("/Users/nicholasmarco/Projects/Simulation/integrated_error/80_obs/trace1/x_results.RDS")
nu_1_true <-  y$B[[1]] %*% t(t(x$nu_true[1,]))
nu_2_true <-  y$B[[1]] %*% t(t(x$nu_true[2,]))
time <- seq(0, 990, 10)
for(i in 1:12){
  dir_i <- paste(dir, "trace", i, "/", sep = "")
  nu_1 <- GetMeanCI_S(dir_i, 50, time, 1)
  nu_2 <- GetMeanCI_S(dir_i, 50, time, 2)

  ## Label Switching
  if(sum(abs(nu_1$CI_50 - nu_1_true)) > sum(abs(nu_1$CI_50 - nu_2_true))){
    nu_i <- nu_1
    nu_1 <- nu_2
    nu_2 <- nu_i
  }
  for(j in 1:100){
    ## Calculate how many points are within the credible interval for mean 1
    if(nu_1$CI_975[j] > nu_1_true[j]){
      if(nu_1$CI_025[j] < nu_1_true[j]){
        counter1 <- counter1 + 1
      }
    }

    ## Calculate how many points are within the credible interval for mean 1
    if(nu_2$CI_975[j] > nu_2_true[j]){
      if(nu_2$CI_025[j] < nu_2_true[j]){
        counter2 <- counter2 + 1
      }
    }
    counterall <- counterall + 1
  }
}


plot(nu_1_true, type = 'l')
lines(nu_1$CI_025, col = "red")
lines(nu_1$CI_975, col = "red")

b <- GetZCI(dir_i, 50)

### Simulation 2

BIC <- matrix(0, 10, 4)
AIC <- matrix(0, 10, 4)
DIC <- matrix(0, 10, 4)

dir <- "/Users/nicholasmarco/Projects/Simulation/Optimal_k/"
time <- seq(0, 990, 10)
time <- rep(list(time), 100)
for(j in 2:5){
  for(i in 1:10){
    print(i)
    print(j)
    x <-  readRDS(paste(dir, j, "_clusters/trace", i,"/x_results.RDS", sep = ""))
    #AIC[i,j-1] <- Model_AIC(paste(dir, j, "_clusters/trace", i, "/", sep = ""), 50, 200, time, x$y_obs)
    #BIC[i,j-1] <- Model_BIC(paste(dir, j, "_clusters/trace", i, "/", sep = ""), 50, 200, time, x$y_obs)
    DIC[i,j-1] <- Model_DIC(paste(dir, j, "_clusters/trace", i, "/", sep = ""), 50, 200, time, x$y_obs)
  }
}

