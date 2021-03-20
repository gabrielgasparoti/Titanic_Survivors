
###### PREVISAO DE SOBREVIVENCIA AO TITANIC ######
# Considerando todos os dados disponiveis e
# criando uma amostra aleatoria para teste

# importando os pacotes
library(tidyverse)
library(kableExtra)
library(factoextra)
library(gridExtra)
library(jtools)
library(fastDummies)
library(caret)
library(pROC)
library(plotly)
library(randomForest)
library(vip)
library(lmtest)


# Importando os dados já processados em python
dir_data <- file.path(dirname(getwd()), 'data')
data <- read.csv( file.path(dir_data, 'full_data.csv') )


# Ajustando dados categóricos
glimpse(data)

data$Survived <- as.factor(data$Survived)
data$Pclass <- as.factor(data$Pclass)
data$Sex <- as.factor(data$Sex)
data$Cabin <- as.factor(data$Cabin)
data$Embarked <- as.factor(data$Embarked)
data$Alone <- as.factor(data$Alone)
data$Title <- as.factor( str_replace(data$Title, ' ', ''))


# Tratamento de variáveis dummies
data_dummy <- dummy_columns(.data = data, 
                            select_columns = c('Pclass', 'Sex', 
                                               'Embarked', 'Title'),
                            remove_most_frequent_dummy = T,
                            remove_selected_columns = T)



# Extraindo dados de treino e teste de forma aleatória
index_train <- sample(length(data_dummy$Survived),
                      size =  length(data_dummy$Survived)*0.7)

train_data <- data_dummy[index_train, ]
test_data <- data_dummy[-index_train, ]

# Validando se as variáveis numericas do conjunto de teste
# estão dentro dos intervalos dos dados de treino
# para evitar extrapolação na predição

summary(train_data)
summary(test_data)




###### MODELO DE REGRESSAO LOGISTICA ######

# Instanciando o modelo glm com os dados de treino
glmodel <- glm(formula = Survived ~ ., 
           data = train_data, 
           family = 'binomial')

# Realizando o procedimento stepwise para descartar variáveis sem significcancia
# estatistica, ou seja, que na presenca das demais nao se mostram significantes
# para explicar o variavel alvo
# condicao = p-valor da variavel menor que 0.05

glmodel <- step(glmodel, k = qchisq(p = 0.05, df = 1, lower.tail = FALSE) )


# Analisando os parâmetros do modelo e o valor do qui-quadrado
summ(glmodel)

# Determinando o qui-quadrado calulado para 8 graus de liberdade
# e nivel de significancia de 5%
# Verificando se o qui-quadrado do modelo é maior que o crítico
# ou seja, se existe significancia estatística
# ou ainda, se existe pelo menos um coeficiente Beta significante
# para explicar o comportamento da variavel alvo
qchisq(p = 0.05, df = 8, lower.tail = F)

# Verificamos que o qui-quadrado do modelo e maior que o crítico
# portanto, existe modelo

# Avaliando o loglik do modelo em relacao ao modelo nulo
# para entender se houve ganho na funcao de maxima verossimilhanca
lrtest(glmodel)


# Verificamos que houve ganho, pois o LL do modelo ficou maior que o LL 
# de um modelo sem os coeficientes calculados, ou seja
# os coeficientes ajudam a maximizar a funcao de maxima verossimilhanca



# Realizando previsão com os dados de treino
train_data$predict_LR <- predict(glmodel, newdata = train_data, 
                                 type = 'response')


# Construindo e visualizando a matriz de confusão para os dados de treino
cm_train <- confusionMatrix(table(train_data$predict_LR >= 0.5, 
                            train_data$Survived == 1)[2:1, 2:1]) 
cm_train


# Criando a curva ROC para os dados de treino
roc_train <- roc(response = train_data$Survived, 
           predictor = glmodel$fitted.values)


# Visualizando a Curva ROC para os dados de treino
ggroc(roc_train, color = 'darkorchid', size = 0.9)+
  geom_segment(aes(x = 0, y =1, xend = 1, yend = 0), 
               color = 'orange', size = 0.9)+
  labs(title = paste('Regressão Logística Binária','\nAUC:',
                     round( roc_train$auc, 3), "|", 
       "GINI:", round( (roc_train$auc - 0.5)/0.5, 3) ),
       subtitle = paste('Eficiência do modelo para os dados de treino:',
                        '\nAcurácia:', round(cm_train$overall[1],3),
                        '\nSensitividade:', round(cm_train$byClass[1], 3),
                        '\nEspecificidade:', round(cm_train$byClass[2], 3)))+
  theme_bw()



# Realizando previsões com a base de teste

# Selecionando apenas variáveis consideradas no modelo
glmodel$coefficients
sample_test <- test_data[, c('Age','SibSp', 'Cabin', 'Alone', 
                             'Pclass_1', 'Pclass_2', 'Sex_female',
                             'Title_Other')]

# Prevendo os resultados
test_data$predict_LR <- predict(glmodel,
                             newdata = sample_test,
                             type = 'response') 



# Construindo e visualizando a matriz de confusão para os dados de teste
cm_test <- confusionMatrix(table(test_data$predict_LR >= 0.5, 
                            test_data$Survived == 1)[2:1, 2:1]) 
cm_test


# Criando a curva ROC para os dados de teste
roc_test <- roc(response = test_data$Survived, 
           predictor = test_data$predict_LR)


# Visualizando a Curva ROC para os dados de treino
ggroc(roc_test, color = 'darkorchid', size = 0.9)+
  geom_segment(aes(x = 0, y =1, xend = 1, yend = 0), 
               color = 'orange', size = 0.9)+
  labs(title = paste('Regressão Logística Binária','\nAUC:',
                     round( roc_test$auc, 3), "|", 
                     "GINI:", round( (roc_test$auc - 0.5)/0.5, 3) ),
       subtitle = paste('Eficiência do modelo para os dados de teste:',
                        '\nAcurácia:', round(cm_test$overall[1],3),
                        '\nSensitividade:', round(cm_test$byClass[1], 3),
                        '\nEspecificidade:', round(cm_test$byClass[2], 3)))+
  theme_bw()


# Salvando o modelo na pasta de modelos
dir_models <- file.path(dirname(getwd()), 'models')
saveRDS(glmodel, file = file.path(dir_models, 'logisticRegression.RDS'))




###### MODELO BASEADO EM ÁRVORES DE DECISÃO ######

# Instanciando o modelo com os dados de treino
# Numero de arvores: 200, pois acima desse numero a diminuicao do erro nao 
# se mostra tão significante
# Numero de variaveis por arvore: 3
# acima disso o modelo sofreu overfitting


RF_model <- randomForest(formula = Survived ~ . - predict_LR,
                         data = train_data, 
                         ntree = 200,
                         mtry = 3,
                         importance = TRUE)


# Visualizando a queda do erro medio
plot(RF_model)


# visualizando as variaveis mais importantes para o modelo
vip(RF_model, aesthetics = list(fill = 'orange',
                                color = 'black',
                                alpha = .8),
    main = 'Title')

# Avaliando o resultado do modelo

# Construindo a matriz de confusao para os dados de treino e teste
# Comparando os resultados com o modelo glm
cm_train_RF <- confusionMatrix(predict(RF_model,
                                       newdata = train_data, 
                                       type = 'class'),
                               train_data$Survived)


cm_test_RF <- confusionMatrix(predict(RF_model,
                                      newdata = test_data,
                                      type = 'class'),
                              test_data$Survived)
cm_train
cm_train_RF

# Para os dados de treino verificamos que o modelo baseado em arvores
# aleatorias ficou mais acurado em relacao ao modelo glm
# 93% de acuracia no RF contra 87% no glm

cm_test
cm_test_RF

# Para os dados de teste verificamos que o modelo baseado em arvores
# aleatorias tambem ficou mais acurado em relacao ao modelo glm
# 88% de acuracia no RF contra 85% no glm


# Alem disso, a sensitividade do modelo RF foi maior que do modelo glm
# 92% em RF contra 77% no glm
# Isso significa que RF acertou uma maior quantidade de observações
# que sobreviveram

# Criacao da curva roc para os dados de treino sobre o modelo de RF
roc_train_RF <- roc(response = train_data$Survived,
                    predictor = predict(RF_model,
                                        newdata = train_data,
                                        type = "prob")[,1])

ggroc(roc_train_RF, color = 'blue', size = .9)+
  geom_segment(aes(x = 0, xend = 1, y = 1, yend = 0), 
               color = 'orange', size = .9)+
  labs(title = paste('Random Forest Classifier\n',
                     'AUC:', round(roc_train_RF$auc, 3),
                     'GINI:', round( (roc_train_RF$auc - 0.5)/0.5, 3)),
       subtitle = paste('Eficiência do modelo para os dados de treino:',
                        '\nAcurácia:', round( cm_train_RF$overall[1], 3),
                        '\nSensitividade:', round (cm_train_RF$byClass[1], 3),
                        '\nEspecificidade:', round( cm_train_RF$byClass[2], 3))) 


# Criacao da curva roc para os dados de teste sobre o modelo de RF
roc_test_RF <- roc(response = test_data$Survived,
                    predictor = predict(RF_model,
                                        newdata = test_data,
                                        type = "prob")[,1])

ggroc(roc_test_RF, color = 'blue', size = .9)+
  geom_segment(aes(x = 0, xend = 1, y = 1, yend = 0), 
               color = 'orange', size = .9)+
  labs(title = paste('Random Forest Classifier\n',
                     'AUC:', round(roc_test_RF$auc, 3),
                     'GINI:', round( (roc_test_RF$auc - 0.5)/0.5, 3)),
       subtitle = paste('Eficiência do modelo para os dados de teste:',
                        '\nAcurácia:', round( cm_test_RF$overall[1], 3),
                        '\nSensitividade:', round (cm_test_RF$byClass[1], 3),
                        '\nEspecificidade:', round( cm_test_RF$byClass[2], 3))) 


# Adicionando as probabilidades estimadas pelo modelo RF nas bases de dados
train_data$predict_RF <- predict(RF_model, 
                                 newdata = train_data, 
                                 type = "prob")[,2]

test_data$predict_RF <- predict(RF_model, 
                                 newdata = test_data, 
                                 type = "prob")[,2]


# Salvando as bases de dados utilizadas no treinamento e validacao dos modelos
write.csv(train_data, file = file.path(dir_data, 'train_data_models.csv'),
          row.names = F)

write.csv(test_data, file = file.path(dir_data, 'test_data_models.csv'), 
          row.names = F)


# Salvando o modelo randomForest
saveRDS(RF_model, file = file.path(dir_models, 'randomForest.RDS'))



################################################################################
