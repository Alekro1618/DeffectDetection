df = read.csv("train.csv")
df["loc"]

activation = function(argument) {
  return (1/(1+exp(-argument)))
}

df[df[23] == "True", 23] = 1
df[df[23] == "False", 23] = 0

df = df[1:10000,]

lowest_error = -1
for (i in 1:10) {
  train = df[-((1000*(i-1)):(1000*i)),]
  validation = df[(1000*(i-1)):(1000*i),]
  w1 = matrix(rnorm(22*22), nrow = 22)
  w2 = matrix(rnorm(22*1), nrow = 1)
  input_layer = integer(22)
  hidden_layer = integer(22)
  output_layer = integer(1)
  output_error = integer(1)
  hidden_error = integer(22)
  lr = 0.001
  for (j in rownames(train)) {
      input_layer = as.numeric(train[j,1:22])
      hidden_layer = activation(w1 %*% input_layer)
      output_layer = activation(w2 %*% hidden_layer)
      
      output_error = (as.numeric(train[j, 23]) - output_layer)*output_layer*(1-output_layer)
      hidden_error = hidden_layer * (1 - hidden_layer) * t(w2) * drop(output_error)
      
      w2 = w2 + lr * drop(output_error) *t(hidden_layer)
      w1 = w1 + lr * hidden_error %*% t(input_layer)
  }
  full_error = 0
  for (j in rownames(validation)) {
      input_layer = as.numeric(validation[j,1:22])
      hidden_layer = activation(w1 %*% input_layer)
      output_layer = activation(w2 %*% hidden_layer)
      full_error = full_error + sum((output_layer - as.numeric(validation[j,23]))^2)
  }
  print(c(full_error, i))
  if (lowest_error < full_error) {
      lowest_error = full_error
      bestw1 = w1
      bestw2 = w2
  }
}
w1 = bestw1
w2 = bestw2
lr = 0.00001
for (j in rownames(df)) {
  input_layer = as.numeric(df[j,1:22])
  hidden_layer = activation(w1 %*% input_layer)
  output_layer = activation(w2 %*% hidden_layer)
  
  output_error = (as.numeric(df[j, 23]) - output_layer)*output_layer*(1-output_layer)
  hidden_error = hidden_layer * (1 - hidden_layer) * t(w2) * drop(output_error)
  
  print(sum((unlist(drop(output_error) *t(hidden_layer)))^2))
  w2 = w2 + lr * drop(output_error) *t(hidden_layer)
  w1 = w1 + lr * hidden_error %*% t(input_layer)
}
full_error = 0
for (j in rownames(df)) {
  input_layer = as.numeric(df[j,1:22])
  hidden_layer = activation(w1 %*% input_layer)
  output_layer = activation(w2 %*% hidden_layer)
  full_error = full_error + sum((output_layer - as.numeric(df[j,23]))^2)
  print(full_error/as.numeric(j))
}
