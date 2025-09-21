# -----------------------------------------------------
# PART 1
# This script implements regression trees attempting to
# improve the prediction on revenue for CheMarket
# 1) Build the regression tree
# 2) Diagram the model
# 3) Evaluate MSE of the model
# 4) d
# -----------------------------------------------------

# -----------------------------------------------------
# 1) Build the regression tree
# -----------------------------------------------------

# Fit the regression tree model (rpart)
set.seed(123)
modelo_arbol <- rpart(Revenue ~ ., data = parte_a, method = "anova")

# Tree model (Caret)
fitControl<-trainControl(method ="cv", number=5)

set.seed(123)

tree_rpart2 <- train(
  formula_tree,
  data=data_train,
  method = "rpart2",
  trControl = fitControl,
  tuneGrid = expand.grid(maxdepth = seq(1,8,1))
)


# -----------------------------------------------------
# 2) Diagram the model
# -----------------------------------------------------

# Rpart
rpart.plot(modelo_arbol)

# Caret


# -----------------------------------------------------
# 3) Evaluate MSE of the model
# -----------------------------------------------------

# Fijamos la semilla
set.seed(2025)

# Crear trainning data (30% of observations)
smp_size <- floor(0.3 * nrow(db))

# Creamos la columna de validación en la db para separar
validacion_ids <- sample(seq_len(nrow(db)), size = smp_size)
db$validacion <- 0
db$validacion[validacion_ids] <- 1

# test and trainning sets
data_test <- db %>% filter(validacion == 1)
data_train <- db %>% filter(validacion == 0)

