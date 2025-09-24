## Parte B - Taller 2 CheMarket INC
## Gianluca Cicco 

# Importar Parte B Taller 1
require(pacman)
p_load(rpart, dplyr, rpart.plot, caret, vtable, devtools, stargazer, causalTree, grf)

db <- readRDS("/Users/gianlucacicco/Desktop/2025-2/Ciencia de Datos/Taller 1/Parte_B.Rds")
setwd("/Users/gianlucacicco/Desktop/2025-2/Ciencia de Datos/Taller 2")

st(db, group = 'sign_up',group.test = TRUE)

# Estadisticas Descriptivas
db <- db |> mutate(ln_revenue = log(Revenue))

# Modelo lineal simple  
mod1 <- lm(ln_revenue ~ sign_up, data = db)
summary(mod1)
mod2 <- lm(ln_revenue ~ sign_up + time_spent + past_sessions + device_type + os_type + is_returning_user, data=db)
summary(mod2)

# Exportar Tabla
stargazer(mod1, mod2, 
          type = "html",
          title = "Efectos del Registro en el Ingreso de CheMarket",
          column.labels = c("Modelo Base", "Modelo Completo"),
          covariate.labels = c("Tratamiento (Registro)", "Tiempo", "Sesiones Pasadas", 
                               "Dispositivo: Movil", "Dispositivo: Tablet", "Sistema: Otro",
                               "Sistema: Windows", "Usuario Recurrente", "Constante"),
          dep.var.labels = "Log(Gasto Mensual)",
          notes = "Errores estándar en paréntesis. *** p<0.01, ** p<0.05, * p<0.1",
          out = "tabla_reglas_completas.html")

# Efectos Heterogeneos
# Modelo 3a - Tecnologia
mod3a <- lm(ln_revenue ~ sign_up + time_spent + past_sessions + device_type*sign_up + os_type*sign_up + is_returning_user, data = db)
# Modelo 3b - Todas las interacciones categoricas
mod3b <- lm(ln_revenue ~ sign_up + time_spent + past_sessions + device_type*sign_up + os_type*sign_up + is_returning_user*sign_up, data = db)

# Exportar Tabla
stargazer(mod1, mod2, mod3a, mod3b,
          type = "html",
          title = "Efectos del Registro en el Ingreso de CheMarket",
          column.labels = c("Modelo Base", "Modelo Completo", "Modelo Efectos Heterogeneos (1)", "Modelo Efectos Heterogeneos (2)"),
          covariate.labels = c("Tratamiento (Registro)", "Tiempo", "Sesiones Pasadas", 
                               "Dispositivo: Movil", "Dispositivo: Tablet", "Sistema: Otro",
                               "Sistema: Windows", "Usuario Recurrente", "Registro: Movil", "Registro: Tablet"
                               , "Registro: Sistema Otro", "Registro: Sistema Windows", "Registro: Usuario Recurrente", "Constante"),
          dep.var.labels = "Log(Gasto Mensual)",
          notes = "Errores estándar en paréntesis. *** p<0.01, ** p<0.05, * p<0.1",
          out = "Tabla_Efectos_Heterogeneos.html")

# Arboles Causales
install_github("susanathey/causalTree")

set.seed(123)
inTrain <- createDataPartition(
  y = db$sign_up,
  p = .5,
  list = FALSE)

db_train <- db[ inTrain,]
db_est <- db[-inTrain,]

ct_unpruned <- honest.causalTree(
  formula = ln_revenue ~ sign_up + time_spent + past_sessions + device_type + os_type + is_returning_user,            
  data = db_train,              
  est_data = db_est,            
  
  treatment = db_train$sign_up,       
  est_treatment = db_est$sign_up,     
  
  split.Rule = "CT",            
  cv.option = "CT",            
  cp = 0,
  
  split.Honest = TRUE,          
  cv.Honest = TRUE,             
  
  minsize = 20,       
  HonestSampleSize = nrow(db_est)) 

# Plot del árbol
par(mar=c(4,4,2,1))
rpart.plot(ct_unpruned, main = "Causal Tree: Efectos Heterogéneos")

# Luego exportar
jpeg("Arbol_Causal_Efectos_Heterogeneos.jpg", width = 1200, height = 800, res = 300, quality = 100)
par(mar=c(4,4,2,1))
rpart.plot(ct_unpruned, main = "Causal Tree: Efectos Heterogéneos de sign_up sobre Revenue")
dev.off()

ct_cptable <- as.data.frame(ct_unpruned$cptable)

# Obtain optimal complexity parameter to prune tree.
selected_cp <- which.min(ct_cptable$xerror)
optim_cp_ct <- ct_cptable[selected_cp, "CP"]
optim_cp_ct
# Como el optimo es 0, nos quedamos con el arbol sin podar.

ct_pruned <- prune(tree = ct_unpruned, cp = optim_cp_ct)

par(mar=c(4,4,2,1))
rpart.plot(ct_pruned, type=1, main = "Causal Tree: Efectos Heterogéneos")

tauhat_ct_est <- predict(ct_pruned, newdata = db_est)

num_leaves <- length(unique(tauhat_ct_est)) 
num_leaves

db_est$leaf <- factor(tauhat_ct_est, labels = seq(num_leaves))

# Run the regression
ols_ct <- lm(as.formula("ln_revenue ~ 0 + leaf + sign_up:leaf"), data = db_est) 
print(as.formula("Y ~ 0 + leaf + sign_up:leaf"))

ols_ct_summary <- summary(ols_ct)
te_summary <- coef(ols_ct_summary)[(num_leaves+1):(2*num_leaves), c("Estimate", "Std. Error")]
te_summary

# Bosques Causales

# Preparar las variables
X <- db %>% 
  select(time_spent, past_sessions, device_type, os_type, is_returning_user) %>%
  model.matrix(~ . - 1, data = .)
Y <- db$ln_revenue
W <- db$sign_up

# Ajustar causal forest (método moderno y robusto)
cf <- causal_forest(X, Y, W, 
                    num.trees = 10000,
                    seed = 123)

# Predecir efectos heterogéneos (CATE)
tau_hat <- predict(cf)$predictions

# Agregar las predicciones a los datos
db$cate_pred <- tau_hat

# Ver estadísticas de las predicciones
summary(tau_hat)

# Ver que tal funciono en las categoricas
# Tipo de dispositivo
db %>% 
  group_by(device_type) %>%
  summarize(
    n = n(),
    tau_hat_promedio = mean(cate_pred),
  )

# Por sistema operativo
db %>% 
  group_by(os_type) %>%
  summarize(
    n = n(),
    tau_hat_promedio = mean(cate_pred),
  )

# Por tipo de usuario
db %>% 
  group_by(is_returning_user) %>%
  summarize(
    n = n(),
    tau_hat_promedio = mean(cate_pred),
  )

# Análisis por grupos de sesiones
db %>% 
  group_by(past_sessions) %>%
  summarize(
    n = n(),
    tau_hat_promedio = mean(cate_pred),
  )

# Distribución de CATEs predichos
ggplot(db, aes(x = cate_pred)) +
  geom_histogram(bins = 40, alpha = 0.7) +
  labs(title = "Distribución de Efectos Heterogéneos Predichos",
       x = "CATE Predicho",
       y = "Frecuencia") +
  theme_minimal()

ggsave("distribucion_efectos_heterogeneos.jpg", 
              width = 12, height = 8, dpi = 300)

# CATEs por tipo de dispositivo
ggplot(db, aes(x = factor(device_type), y = cate_pred, fill = factor(device_type))) +
  geom_boxplot() +
  labs(title = "CATEs Predichos por Tipo de Dispositivo",
       x = "Tipo de Dispositivo",
       y = "CATE Predicho") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("Boxplot_CATE_Dispositivo.jpg", 
       width = 12, height = 8, dpi = 300)

# CATEs por sistema operativo
ggplot(db, aes(x = factor(os_type), y = cate_pred, fill = factor(os_type))) +
  geom_boxplot() +
  labs(title = "CATEs Predichos por Sistema Operativo",
       x = "Sistema Operativo",
       y = "CATE Predicho") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("Boxplot_CATE_Sistema.jpg", 
       width = 12, height = 8, dpi = 300)

# CATEs por tipo de usuario
ggplot(db, aes(x = factor(is_returning_user), y = cate_pred, fill = factor(is_returning_user))) +
  geom_boxplot() +
  scale_x_discrete(labels = c("Usuario Nuevo", "Usuario Recurrente")) +
  labs(title = "CATEs Predichos por Tipo de Usuario",
       x = "Tipo de Usuario",
       y = "CATE Predicho") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("Boxplot_CATE_UsuarioRecurrente.jpg", 
       width = 12, height = 8, dpi = 300)
# El hecho que se crucen los boxplot, indica que si hay subgrupos, los cuales vamos a observar con la diferencia entre grupos, para ello 
# primero sacamos todas las posibles combinaciones y a partir de ahi señalamos si hay diferencias

# Conocer todas las interacciones posibles
vars <- c("time_spent", "past_sessions", "device_type", "os_type", "is_returning_user")
combinaciones <- combn(vars, 2, simplify = FALSE)

for(i in 1:length(combinaciones)) {
  var1 <- combinaciones[[i]][1]
  var2 <- combinaciones[[i]][2]
  
  cat(i, ":", var1, "×", var2, "\n")
}

# 1 : time_spent × past_sessions 
# 2 : time_spent × device_type 
# 3 : time_spent × os_type 
# 4 : time_spent × is_returning_user 
# 5 : past_sessions × device_type 
# 6 : past_sessions × os_type 
# 7 : past_sessions × is_returning_user 
# 8 : device_type × os_type 
# 9 : device_type × is_returning_user 
# 10 : os_type × is_returning_user 

## Efectos Heterogéneos - Todas las Interacciones

# 1: time_spent × past_sessions
tau_hat_summ_1 <- db |> 
  group_by(time_spent, past_sessions) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_1 <- ggplot(tau_hat_summ_1) + 
  geom_point(aes(x = time_spent, y = tau_hat, color = factor(past_sessions))) + 
  labs(x = "Tiempo en Sesión", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Sesiones Pasadas") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_time_past.jpg", 
       plot = efectotratamiento_1,
       width = 12, height = 8, dpi = 300)

# 2: time_spent × device_type
tau_hat_summ_2 <- db |> 
  group_by(device_type, time_spent) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_2 <- ggplot(tau_hat_summ_2) + 
  geom_point(aes(x = time_spent, y = tau_hat, color = device_type)) + 
  labs(x = "Tiempo en Sesión", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Tipo de Dispositivo") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_time_device.jpg", 
       plot = efectotratamiento_2,
       width = 12, height = 8, dpi = 300)

# 3: time_spent × os_type
tau_hat_summ_3 <- db |> 
  group_by(os_type, time_spent) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_3 <- ggplot(tau_hat_summ_3) + 
  geom_point(aes(x = time_spent, y = tau_hat, color = os_type)) + 
  labs(x = "Tiempo en Sesión", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Sistema Operativo") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_time_os.jpg", 
       plot = efectotratamiento_3,
       width = 12, height = 8, dpi = 300)

# 4: time_spent × is_returning_user
tau_hat_summ_4 <- db |> 
  group_by(is_returning_user, time_spent) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_4 <- ggplot(tau_hat_summ_4) + 
  geom_point(aes(x = time_spent, y = tau_hat, color = is_returning_user)) + 
  labs(x = "Tiempo en Sesión", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Usuario Recurrente") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_time_returning.jpg", 
       plot = efectotratamiento_4,
       width = 12, height = 8, dpi = 300)

# 5: past_sessions × device_type
tau_hat_summ_5 <- db |> 
  group_by(device_type, past_sessions) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_5 <- ggplot(tau_hat_summ_5) + 
  geom_point(aes(x = past_sessions, y = tau_hat, color = device_type), size=3) + 
  labs(x = "Sesiones Pasadas", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Tipo de Dispositivo") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_sessions_device.jpg", 
       plot = efectotratamiento_5,
       width = 12, height = 8, dpi = 300)

# 6: past_sessions × os_type
tau_hat_summ_6 <- db |> 
  group_by(os_type, past_sessions) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_6 <- ggplot(tau_hat_summ_6) + 
  geom_point(aes(x = past_sessions, y = tau_hat, color = os_type), size=3) + 
  labs(x = "Sesiones Pasadas", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Sistema Operativo") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_sessions_os.jpg", 
       plot = efectotratamiento_6,
       width = 12, height = 8, dpi = 300)

# 7: past_sessions × is_returning_user
tau_hat_summ_7 <- db |> 
  group_by(is_returning_user, past_sessions) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_7 <- ggplot(tau_hat_summ_7) + 
  geom_point(aes(x = past_sessions, y = tau_hat, color = is_returning_user), size=3) + 
  labs(x = "Sesiones Pasadas", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Usuario Recurrente") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_sessions_returning.jpg", 
       plot = efectotratamiento_7,
       width = 12, height = 8, dpi = 300)

# 8: device_type × os_type
tau_hat_summ_8 <- db |> 
  group_by(device_type, os_type) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_8 <- ggplot(tau_hat_summ_8) + 
  geom_point(aes(x = device_type, y = tau_hat, color = os_type), size = 4) + 
  labs(x = "Tipo de Dispositivo", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Sistema Operativo") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_device_os.jpg", 
       plot = efectotratamiento_8,
       width = 12, height = 8, dpi = 300)

# 9: device_type × is_returning_user
tau_hat_summ_9 <- db |> 
  group_by(device_type, is_returning_user) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_9 <- ggplot(tau_hat_summ_9) + 
  geom_point(aes(x = device_type, y = tau_hat, color = factor(is_returning_user)), size = 4) + 
  labs(x = "Tipo de Dispositivo", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Usuario Recurrente") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_device_returning.jpg", 
       plot = efectotratamiento_9,
       width = 12, height = 8, dpi = 300)

# 10: os_type × is_returning_user
tau_hat_summ_10 <- db |> 
  group_by(os_type, is_returning_user) |>
  summarize(
    tau_hat = mean(cate_pred),
    .groups = 'drop'
  )

efectotratamiento_10 <- ggplot(tau_hat_summ_10) + 
  geom_point(aes(x = os_type, y = tau_hat, color = factor(is_returning_user)), size = 4) + 
  labs(x = "Sistema Operativo", 
       y = "Efecto del Tratamiento Estimado", 
       color = "Usuario Recurrente") +
  theme_minimal(base_size = 16)

ggsave("efecto_tratamiento_os_returning.jpg", 
       plot = efectotratamiento_10,
       width = 12, height = 8, dpi = 300)


