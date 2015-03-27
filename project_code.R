library(caret)
library(randomForest)

org.training.data <- read.csv("~/Downloads/pml-training.csv")
training.data <- org.training.data[,-1:-7]
training.data[,-ncol(training.data)] <- apply(training.data[,-ncol(training.data)], 2, function(x) as.numeric(x))
num.nas <- apply(training.data, 2, function(x) sum(is.na(x)))

cols.2.remove <- num.nas >= 19216
clean.data <- training.data[,!cols.2.remove]
clean.vals <- clean.data[,ncol(clean.data)]
clean.predictors <- clean.data[,-ncol(clean.data)]

dumbbell.cols <- grepl("dumbbell",names(clean.predictors))
clean.predictors <- clean.predictors[,!(dumbbell.cols)]

colExclude <- findCorrelation(cor(clean.predictors), cutoff=0.9, verbose=F)
clean.predictors <- clean.predictors[,-(colExclude)]

table(names(clean.predictors))
set.seed(861989)
model.fit <- tuneRF(x=clean.predictors, y=clean.vals, ntreeTry=100, stepFactor=1.5, doBest=T, importance=T)

err.rates <- model.fit$err.rate*100
#color palette:  black, orange,	sky blue, bluish green, yellow, blue, vermillion, pink
cbbPalette <- c(black="#000000", orange="#E69F00", cyan="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", red="#D55E00", pink="#CC79A7")

plot(err.rates[,"OOB"],type="l",log="y",ylim=c(0.05,10), xlab="Trees", ylab="Error Rate (%)", main="Out of the Box and Class Error Rates")
lines(err.rates[,"A"], lty=2, col=cbbPalette["red"])
lines(err.rates[,"B"], lty=3, col=cbbPalette["green"])
lines(err.rates[,"C"], lty=4, col=cbbPalette["orange"])
lines(err.rates[,"D"], lty=5, col=cbbPalette["cyan"])
lines(err.rates[,"E"], lty=1, col=cbbPalette["pink"])
legend(x="topright", legend=colnames(err.rates), lty=c(1,2,3,4,5,1), col=c(cbbPalette["black"], cbbPalette["red"], cbbPalette["green"], cbbPalette["orange"], cbbPalette["cyan"], cbbPalette["pink"]))

varImpPlot(model.fit, main="Variable Importance Plots")

org.testing.data <- read.csv("~/Downloads/pml-testing.csv")
predictor.cols <- colnames(clean.data)[-length(colnames(clean.data))]
testing.data <- org.testing.data[,c(predictor.cols, "problem_id")]

preds <- predict(model.fit, testing.data)
preds