
accuracy_fun <- function(truth, pred){
  sum(diag(table(truth, pred)))/sum(table(truth, pred, useNA = "always"))
}

rmse_fun <- function(truth, pred){
  sqrt(mean((truth - pred)^2, na.rm = TRUE))
}
