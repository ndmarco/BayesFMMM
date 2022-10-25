library(BayesFMMM)
library(eegkit)
library(gridExtra)
library(grDevices)
library(ggplot2)

################################################
## Change relevant directories before running ##
################################################

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

load("/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/pa.dat.Rdata")
chan_id <- c('FP1', 'FP2','F9','F7','F3','FZ','F4','F8','F10','T9','T7',
             'C3','CZ','C4','T8','T10','P9','P7','P3','PZ','P4','P8','P10','O1','O2')
chan_id_sub <- c('F5', 'F6', 'T7', 'CZ', 'T8', 'PZ')
## Get coordinates
data("eegcoord")
electrode_loc <- eegcoord[-c(1,2),]
electrode_loc <- electrode_loc[-52,]
#electrode_loc <- eegcoord[match(chan_id_sub, rownames(eegcoord)),]
electrode_loc <- eegdense
func <- 1
time <- matrix(0, nrow = length(func) * nrow(electrode_loc), ncol = 3)
counter = 1
data("eegdense")
for(i in 1:nrow(electrode_loc)){
  for(k in 1:length(func)){
    time[counter,1] <- electrode_loc$xproj[i]
    time[counter,2] <- electrode_loc$yproj[i]
    time[counter,3] <- func[k]
    counter = counter + 1
  }
}
## Set Hyperparameters
dir <- "/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/trace/"
n_files <- 19
k <- 2
basis_degree <- c(2,2,2)
boundary_knots <- matrix(c(-14, 14, -12, 12.5, 6, 14), nrow = 3, byrow = T)
internal_knots1 <- c(-7, 0, 7)
internal_knots2 <- c(-6, 0, 6)
internal_knots3 <- c(8.33, 11.66)
internal_knots <- list(internal_knots1, internal_knots2, internal_knots3)

## Get CI for mean function
CI1 <- HDFMeanCI(dir, n_files, time, basis_degree, boundary_knots, internal_knots, 1)
CI2 <- HDFMeanCI(dir, n_files, time, basis_degree, boundary_knots, internal_knots, 2)
colfunc<-colorRampPalette()
color.gradient <- function(x, colors=c("royalblue", "red"), colsteps=100) {
  return( colorRampPalette(colors) (colsteps) [ findInterval(x, seq(min(x),max(x), length.out=colsteps)) ] )
}
ph <- c(CI1$CI_50, CI2$CI_50)
cols <- color.gradient(ph)

jpeg("/Users/user/Desktop/images/")
for(i in 1:length(func)){
  par(mfrow = c(1,2))
  # par(fig=c(0,0.45,0.4,1))
  eegcapdense(plotlabels = F, col.point = cols[((i-1)*nrow(eegdense) + 1):(i*nrow(eegdense))], main = "Cluster 1")
  # par(fig=c(0.55,1,0.4,1), new = T)
  eegcapdense(plotlabels = F, col.point = cols[((i-1)*nrow(eegdense) + nrow(time) + 1):(i*nrow(eegdense)+ nrow(time))], main = "Cluster 2")
  # par(fig=c(0,1,0,0.35), new = T)
  # plot(rep(1,100),col=colorRampPalette(c("royalblue", "red"))(100), pch=19,cex=2, xaxt='n',xlab = NA, ylab  = NA, yaxt='n', main = paste(func[i], "Hz"))
}
dev.off()

## Get CI for Z
Z <- GetZCI(dir, n_files)

## Get covariance functions
time[,3] <- 6
time1 <- time[1:500,]
time2 <- time[501:977,]

Cov1 <- HDFCovCI(dir, n_files, n_MCMC = 100, time1, time1, basis_degree, boundary_knots, internal_knots, 1,1)
ph <- diag(Cov1$CI_50)
Cov1 <- HDFCovCI(dir, n_files, n_MCMC = 100, time2, time2, basis_degree, boundary_knots, internal_knots, 1,1)
ph <- c(ph, diag(Cov1$CI_50))

#saveRDS(Cov1, "/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/cov1_6hz.RDS")
Cov2 <- HDFCovCI(dir, n_files, n_MCMC = 100, time1, time1, basis_degree, boundary_knots, internal_knots, 2,2)
ph <- c(ph, diag(Cov2$CI_50))
Cov2 <- HDFCovCI(dir, n_files, n_MCMC = 100, time2, time2, basis_degree, boundary_knots, internal_knots, 2,2)
ph <- c(ph, diag(Cov2$CI_50))

cols <- color.gradient(ph)

par(mfrow = c(1,2))
# par(fig=c(0.55,1,0.4,1), new = T)
eegcapdense(plotlabels = F, col.point = cols[978:1954])
# par(fig=c(0,0.45,0.4,1))
eegcapdense(plotlabels = F, col.point = cols[1:977])


#saveRDS(Cov2, "/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/cov2_6hz.RDS")
time[,3] <- 10
Cov1 <- HDFCovCI(dir, n_files, n_MCMC = 100, time, time, basis_degree, boundary_knots, internal_knots, 1,1)
#saveRDS(Cov1, "/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/cov1_10hz.RDS")
Cov2 <- HDFCovCI(dir, n_files, n_MCMC = 100, time, time, basis_degree, boundary_knots, internal_knots, 2,2)
#saveRDS(Cov2, "/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/cov2_10hz.RDS")

ph <- c(diag(Cov1$CI_50), diag(Cov2$CI_50))
cols <- color.gradient(ph)

#
# make.mov <- function(){
#   unlink("/Users/user/Desktop/images/ASD.mpg")
#   system("convert -delay 0.5 /Users/user/Desktop/images/ASD*.jpg /Users/user/Desktop/images/ASD.mpg")
# }


x <- matrix(0, nrow = length(CI1$CI_50), ncol = 3)
x <- as.data.frame(x)
x$V1 <- CI1$CI_50
for(i in 1:6){
  x[((i-1)*length(func) + 1):(i * length(func)),2] <- rownames(electrode_loc)[i]
  x[((i-1)*length(func) + 1):(i * length(func)),3] <- func
}
x$V2 <- as.factor(x$V2)
colnames(x) <- c("Power", "Electrode", "Frequency (Hz)")
p <- ggplot(data= x, aes(x = `Frequency (Hz)`, y = Power, color=Electrode )) + geom_line() + ggtitle("Feature 2") + theme_classic() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                                panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
                                                                                                plot.title = element_text(hjust = 0.5))

x <- matrix(0, nrow = length(CI1$CI_50), ncol = 3)
x <- as.data.frame(x)
x$V1 <- CI2$CI_50
for(i in 1:6){
  x[((i-1)*length(func) + 1):(i * length(func)),2] <- rownames(electrode_loc)[i]
  x[((i-1)*length(func) + 1):(i * length(func)),3] <- func
}
x$V2 <- as.factor(x$V2)
colnames(x) <- c("Power", "Electrode", "Frequency (Hz)")
p2 <- ggplot(data= x, aes(x = `Frequency (Hz)`, y = Power, color=Electrode )) + geom_line() + ggtitle("Feature 1") + theme_classic() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                                                                            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
                                                                                                                                            plot.title = element_text(hjust = 0.5))


grid.arrange(p2, p, ncol = 2)
Z <- ZCI(dir, n_files)
demDat <- read.csv(file='/Users/user/Box Sync/BayesFMMM_Supporting_Files/ASD_multivariate/demographic_data.csv', header = TRUE)
colnames(demDat) <- c("ID", "Gender", "Age", "Group", "VIQ", "NVIQ")
demDat <- demDat[which(demDat$ID %in% subj_id), ]
data_Z <- data.frame("Cluster 1" = Z$CI_50[,2], "Clinical Diagnosis" = demDat$Group)
data_Z$Clinical.Diagnosis[data_Z$Clinical.Diagnosis == 2] <- "ASD"
data_Z$Clinical.Diagnosis[data_Z$Clinical.Diagnosis == 1] <- "TD"
ggplot(data= data_Z, aes(x = `Cluster.1` , y = Clinical.Diagnosis)) + geom_violin(trim = F, xlim = c(0,1)) + geom_point() + xlab("Feature 1") + ylab("Clinical Diagnosis") +
  stat_summary(
    geom = "point",
    fun.x = "mean",
    col = "black",
    size = 3,
    shape = 24,
    fill = "red")+ xlim(c(0,1)) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
                                        panel.background = element_blank(),axis.line = element_line(colour = "black"),
                                        plot.title = element_text(hjust = 0.5))
