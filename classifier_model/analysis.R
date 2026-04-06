# analysis.R
# This script computes the LinearSVC predictions based on input features and model weights.

# Load required features and model parameters
features <- read.csv("temp/features.csv")
coefs <- read.csv("models/coef.csv")
intercepts <- read.csv("models/intercept.csv")
classes <- read.csv("models/classes.csv", stringsAsFactors=FALSE)

# Convert to matrices for vectorized operations
X <- as.matrix(features)
W <- as.matrix(coefs)
b <- as.numeric(intercepts$intercept)

# Compute decision function scores: X * W^T + b
# Dim: (n_samples, n_features) %*% (n_features, n_classes) = (n_samples, n_classes)
scores <- X %*% t(W)

# Add intercepts to each class score
for (i in seq_along(b)) {
    scores[, i] <- scores[, i] + b[i]
}

# Find the index of the maximum score for each sample
max_idx <- apply(scores, 1, which.max)

# Map the index to the original class name
predicted_class <- classes$class[max_idx]

# Write out the predicted class
writeLines(predicted_class, "temp/output.txt")
