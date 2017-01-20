# Maximum likelihood estimation of the mean and covariante of a multivariate 
# normal distribution from data in which some observations may be missing
# (with at most one missing observation per case), using EM.
#
# Arguments:
#
#    X      Data matrix (rows are iid observations)
#    iters  Number of iterations of EM to do
#    debug  Level of debug output (0 = none, 1 = parameters, 2 = filled in data)
#
# Value returned:  A list with elements mu and sigma giving the MLE.

norm_em2 <- function (X, iters, maxit, tol, output)
{
  counter <- 0
  if (any(rowSums(is.na(X))>1))
    stop("This function can handle at most one missing value per case")
  
  n <- nrow(X)
  p <- ncol(X)
  
  # Find initial estimates, using simple averages with missing observations
  # omitted.  The initial covariance estimate is diagonal.
  
  mu <- colMeans(X,na.rm=TRUE)*0.8
  sigma <- diag(apply(X,2,var,na.rm=TRUE))*0.8
  
  
  
  # Update the parameter estimates with iterations of the EM algorithm.
  
  filled_in_X <- X
  while (counter <= maxit) { 
    counter <- counter + 1
    cat("iter = ", counter, "Mu = ", mu, "\n")
    # E Step:
    #
    # Compute a filled in version of X, with missing values set to their
    # conditional mean given other values and current parameter estimates,
    # along with the total extra variance due to uncertainty in the
    # missing values.
    
    
    extra_var <- numeric(p)
    for (i in 1:n) {
      x <- X[i,]
      na <- is.na(x)
      if (any(na)) {
        sigma_inv <- solve(sigma[!na,!na])
        cov_vec <- sigma[na,!na]
        filled_in_X[i,na] <- 
          mu[na] + cov_vec %*% sigma_inv %*% (x[!na] - mu[!na])
        extra_var[na] <- extra_var[na] + 
          sigma[na,na] - cov_vec %*% sigma_inv %*% cov_vec
      }
    }
    
    # M Step:
    #
    # Find new parameter estimates from the filled in version of X, by
    # taking simple averages over the filled in data.
    
    mu.new <- colMeans(filled_in_X)
    Y <- t(filled_in_X) - mu.new
    sigma.new<- (Y %*% t(Y) + diag(extra_var)) / n
    if (max(abs((sigma.new - sigma)/sigma.new)) < tol  ) {
      cat("\nSuccessfully Converged\n")
      write.csv(filled_in_X, file=output)
      return(mu.new)
    } else {
      mu <- mu.new
      sigma <- sigma.new
    }
  }
  
  #list(mu=mu,sigma=sigma)
  print("\n Failed to converge\n")
  return(mu)
}

    
    norm_em <- function (X, iters, maxit, tol, output)
    {
      counter <- 0
      if (any(rowSums(is.na(X))>1))
        stop("This function can handle at most one missing value per case")
      
      n <- nrow(X)
      p <- ncol(X)
      
      # Find initial estimates, using simple averages with missing observations
      # omitted.  The initial covariance estimate is diagonal.
      
      mu <- colMeans(X,na.rm=TRUE)
      sigma <- diag(apply(X,2,var,na.rm=TRUE))
      
      
      
      # Update the parameter estimates with iterations of the EM algorithm.
      
      filled_in_X <- X
      while (counter <= maxit) { 
        counter <- counter + 1
        cat("iter = ", counter, "Mu = ", mu, "\n")
        # E Step:
        #
        # Compute a filled in version of X, with missing values set to their
        # conditional mean given other values and current parameter estimates,
        # along with the total extra variance due to uncertainty in the
        # missing values.
        
        
        extra_var <- numeric(p)
        for (i in 1:n) {
          x <- X[i,]
          na <- is.na(x)
          if (any(na)) {
            sigma_inv <- solve(sigma[!na,!na])
            cov_vec <- sigma[na,!na]
            filled_in_X[i,na] <- 
              mu[na] + cov_vec %*% sigma_inv %*% (x[!na] - mu[!na])
            extra_var[na] <- extra_var[na] + 
              sigma[na,na] - cov_vec %*% sigma_inv %*% cov_vec
          }
        }
        
        # M Step:
        #
        # Find new parameter estimates from the filled in version of X, by
        # taking simple averages over the filled in data.
        
        mu.new <- colMeans(filled_in_X)
        Y <- t(filled_in_X) - mu.new
        sigma.new<- (Y %*% t(Y) + diag(extra_var)) / n
        if (max(abs((sigma.new - sigma)/sigma.new)) < tol  ) {
          cat("\nSuccessfully Converged\n")
          write.csv(filled_in_X, file=output)
          return(mu.new)
        } else {
          mu <- mu.new
          sigma <- sigma.new
        }
  }
  
  #list(mu=mu,sigma=sigma)
  print("\n Failed to converge\n")
 return(mu)
}


# Test on a small dataset.

cat("TEST ON A SMALL DATASET\n")

X <- rbind (c( 1.0,-1.1,  NA),
            c(-0.3,  NA, 2.2),
            c( 1.3,-0.2, 0.1),
            c(  NA, 2.1,-0.3)
)

cat("\nCovariance estimate from pairwise complete observations:\n")
print(round(cov(X,use="pairwise.complete.obs"),3))

cat("\nResult of maximum likelihood estimation with EM:\n")
norm_em(X, 30, 30, 10^-3,  )


# Test on a large simulated dataset to check that the results are correct.

n <- 10000

cat("\nTEST ON A LARGE SIMULATED DATASET (n =",n,")\n")

set.seed(1)

S <- matrix(rnorm(3*4),3,4)
S <- S %*% t(S)
cat("\nTrue covariance matrix:\n")
print(round(S,3))

full_X <- matrix(rnorm(n*3),n,3) %*% chol(S)
cat("\nSample covariance of complete data:\n")
print(round(cov(full_X),3))

X <- full_X
for (i in 1:n) X[i,sample(1:3,1)] <- NA
cat("\nCovariance estimate from pairwise complete observations:\n")
print(round(cov(X,use="pairwise.complete.obs"),3))

cat("\nResult of maximum likelihood estimation with EM:\n")
print(norm_em(X,30,debug=1))