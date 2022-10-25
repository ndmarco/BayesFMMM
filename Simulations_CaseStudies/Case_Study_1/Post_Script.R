library(BayesFMMM)

#################################################
## Change relevant directories  before running ##
#################################################

### Peak alpha data
library(pracma)
library(gridExtra)
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
chan_id <- c('FP1', 'FP2','F9','F7','F3','Fz','F4','F8','F10','T9','T7',
             'C3','CZ','C4','T8','T10','P9','P7','P3','PZ','P4','P8','P10','O1','O2')

# Demographic Data
demDat <- read.csv(file='/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/demographic_data.csv', header = TRUE)
colnames(demDat) <- c("ID", "Gender", "Age", "Group", "VIQ", "NVIQ")
demDat <- demDat[which(demDat$ID %in% subj_id), ]

# Peak Alpha Data
load("/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/pa.dat.Rdata")
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
time <- rep(list(time), 97)
basis_degree <- 3
n_eigen <- 3
boundary_knots <- c(6, 14)
internal_knots <- c(7.6, 9.2, 10.8, 12.4)
time <- seq(6, 14, 0.01)
dir <- "/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/Univariate/trace3/"
### get credible intervals for mean
mean_1 <- FMeanCI(dir, 50, time, basis_degree, boundary_knots, internal_knots, 3)
plot(time,mean_1$CI_50, type = 'l')
lines(time, mean_1$CI_Lower, col = "red")
lines(time, mean_1$CI_Upper, col = "red")

mean_1s <- FMeanCI(dir, 50, time, basis_degree, boundary_knots, internal_knots, 2, simultaneous = T)

mean_2 <- FMeanCI(dir, 50, time, basis_degree, boundary_knots, internal_knots, 1)
plot(time,mean_2$CI_50, type = 'l', xlab = "Frequency (Hz)", ylab = "Power", ylim = c(0, 0.4))
lines(time,mean_2$CI_025, col = "red")
lines(time,mean_2$CI_975, col = "red")

mean_2s <- FMeanCI(dir, 50, time, basis_degree, boundary_knots, internal_knots, 1, simultaneous = T)


predframe <- data.frame(freq = time,
                        median=mean_1$CI_50,lwr=mean_1$CI_Lower,upr=mean_1$CI_Upper, lwr_s = mean_1s$CI_Lower, upr_s = mean_1s$CI_Upper)
p1 <- ggplot(predframe, aes(freq, median))+
  geom_line(col = "blue")+
  geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3, fill = "black") + geom_ribbon(data=predframe,aes(ymin=lwr_s,ymax=upr_s),alpha=0.4, fill = "dark grey")  + ylab("Power") +
  xlab("Frequency (Hz)") + ylim(c(-0.1, 0.65)) + xlim(c(6,14)) + ggtitle("Mean 1") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))

predframe <- data.frame(freq = time,
                        median=mean_2$CI_50,lwr=mean_2$CI_Lower,upr=mean_2$CI_Upper, lwr_s = mean_2s$CI_Lower, upr_s = mean_2s$CI_Upper)
p2<- ggplot(predframe, aes(freq, median))+
  geom_line(col = "blue")+
  geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3, fill = "black") + geom_ribbon(data=predframe,aes(ymin=lwr_s,ymax=upr_s),alpha=0.4, fill = "dark grey") + ylab("Power")+
  xlab("Frequency (Hz)") + ylim(c(0, 0.35)) + xlim(c(6,14)) + ggtitle("Mean 2") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))

grid.arrange(p1, p2, ncol = 2)


Z_post <- ZCI(dir, 50)
data_Z <- data.frame("Cluster 1" = Z_post$CI_50[,2], "Clinical Diagnosis" = demDat$Group)
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

data_VIQ <- data_Z <- data.frame("Cluster 1" = Z_post$CI_50[,3], "VIQ" = demDat$VIQ)
p1 <- ggplot(data= data_Z, aes(x = `Cluster.1` , y = VIQ)) + geom_point() + xlab("Cluster 1") + ylab("Verbal IQ") +
  geom_smooth(method='lm', colour = "red") + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                   panel.background = element_blank(),axis.line = element_line(colour = "black"),
                                                   plot.title = element_text(hjust = 0.5))
data_NVIQ <- data_Z <- data.frame("Cluster 1" = Z_post$CI_50[,3], "NVIQ" = demDat$NVIQ)
p2 <- ggplot(data= data_Z, aes(x = `Cluster.1` , y = NVIQ)) + geom_point() + xlab("Cluster 1") + ylab("Nonverbal IQ") +
  geom_smooth(method='lm', colour = "red") + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                                   panel.background = element_blank(),axis.line = element_line(colour = "black"),
                                                   plot.title = element_text(hjust = 0.5))

grid.arrange(p1, p2, ncol = 2)
