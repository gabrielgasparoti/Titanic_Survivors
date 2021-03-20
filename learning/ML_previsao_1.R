
# PREVISAO DE SOBREVIVENCIA AO TITANIC

# TREINAMENTO DO MODELO DE REGRESSAO LOGISTICA SOBRE OS DADOS DE TREINO


# Neste script treinamos o modelo com base nos dados de treino 
# Fizemos previsões para os dados de teste

library(tidyverse)
library(kableExtra)
library(factoextra)
library(gridExtra)
library(jtools)
library(fastDummies)
library(caret)
library(pROC)
library(plotly)


# Importando os dados já processados em python
dir_data <- file.path(dirname(getwd()), 'data')
treino <- read.csv( file.path(dir_data, 'processed_train_data.csv') )

# Ajustando dados categóricos
glimpse(treino)

treino$Survived <- as.factor(treino$Survived)
treino$Pclass <- as.factor(treino$Pclass)
treino$Sex <- as.factor(treino$Sex)
treino$Cabin <- as.factor(treino$Cabin)
treino$Embarked <- as.factor(treino$Embarked)
treino$Alone <- as.factor(treino$Alone)
treino$Title <- as.factor( str_replace(treino$Title, ' ', ''))

# Criando variáveis dummies
treino_dummy <- dummy_columns(.data = treino, 
                            select_columns = c('Pclass', 'Sex', 
                                               'Embarked', 'Title'),
                            remove_most_frequent_dummy = T,
                            remove_selected_columns = T)

# Instanciando o modelo glm
glmodel <- glm(formula = Survived ~ ., 
           data = treino_dummy, 
           family = 'binomial')

# Realizando o procedimento stepwise para descartar variáveis sem significância
glmodel <- step(glmodel, k = qchisq(p = 0.05, df = 1, lower.tail = FALSE) )

# Analisando os parâmetros do modelo
summ(glmodel)

# Realizando previsão com os dados de treino
treino$Predict <- predict(glmodel, newdata = treino_dummy, type = 'response')

# Construindo a matriz de confusão
cm <- confusionMatrix(table(predict(glmodel, type = "response") >= 0.5, 
                            treino_dummy$Survived == 1)[2:1, 2:1]) 

# Visualizando a matriz
cm

# Criando a curva ROC
roc <- roc(response = treino$Survived, 
           predictor = glmodel$fitted.values)

# Visualizando a Curva ROC
ggroc(roc, color = 'darkorchid', size = 0.9)+
  geom_segment(aes(x = 0, y =1, xend = 1, yend = 0), 
               color = 'orange', size = 0.9)+
  labs(title = paste('Regressão Logística Binária','\nAUC:', round( roc$auc, 3), "|", 
       "GINI:", round( (roc$auc - 0.5)/0.5, 3) ),
       subtitle = paste('Eficiência do modelo para os dados de treino:',
                        '\nAcurácia:', round(cm$overall[1],3),
                        '\nSensitividade:', round(cm$byClass[1], 3),
                        '\nEspecificidade:', round(cm$byClass[2], 3)))+
  theme_bw()


# Importando a base de teste já processa em python
teste <- read.csv( file.path (dir_data, 'processed_test_data.csv') )

# Ajustando o tipo de dado
teste$Pclass <- as.factor(teste$Pclass)
teste$Sex <- as.factor(teste$Sex)
teste$Cabin <- as.factor(teste$Cabin)
teste$Embarked <- as.factor(teste$Embarked)
teste$Alone <- as.factor(teste$Alone)
teste$Title <- as.factor( str_replace(teste$Title, ' ', ''))

# Excluindo dados fora do intervalo de treino para evitar extrapolação
summary(treino)
summary(teste)

teste <- subset(teste, teste$Age >= 2.5)
teste <- subset(teste, teste$Age <= 54.5)
teste <- subset(teste, teste$Fare <= 65.63)

# Criando a base de dummies
teste_dummies <- dummy_columns(.data = teste,
                               select_columns = c('Pclass', 'Sex', 
                                                  'Embarked', 'Title'),
                               remove_selected_columns = T,
                               remove_first_dummy = F,
                               remove_most_frequent_dummy = F)

# Selecionando apenas variáveis consideradas no modelo
glmodel$coefficients
teste_dummies2 <- teste_dummies[, c('Age','SibSp', 'Cabin', 'Alone',
                                    'Pclass_1', 'Pclass_2', 'Sex_female',
                                    'Title_Other')]
# Prevendo os resultados
teste$resp <- predict(glmodel,
                      newdata = teste_dummies2,
                      type = 'response') >= .5

# Ajuste na visualização da variável resposta
teste$resp <- str_replace(teste$resp, 'TRUE', '1')
teste$resp <- str_replace(teste$resp, 'FALSE', '0')
teste$resp <- as.factor(teste$resp)

# Salvando a base de teste com a previsão da variável resposta
# Será concatenada em python com a base de treino para uma terceira análise
write.csv(teste, file = file.path(dir_data, 'teste_predict.csv'), row.names = F )
