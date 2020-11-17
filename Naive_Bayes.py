from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, KFold
from numpy import mean
import pandas as pd

#Carrega o Data Set
cancer = load_breast_cancer()

# Cria o dataframe
df_cancer = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df_cancer['class'] = cancer.target

#Divide a coluna da classe das demais
Y = df_cancer.pop('class')
X = df_cancer

#Utiliza validação cruzada com k=10
modelo=GaussianNB()
cv = KFold(n_splits=10, shuffle=True)
scoring=['accuracy','precision', 'recall', 'f1']
scores = cross_validate(modelo, X,Y, scoring=scoring, cv=cv)

#Retira parte indesejada do resultado(tempo para treino e para teste)
scores.pop('fit_time')
scores.pop('score_time')

# Printa a acurácia e a acurácia média
print('\nAccurácia de cada Fold da Validação Cruzada:' , scores['test_accuracy'])
print('Accurácia Média:' , mean(scores['test_accuracy']))

# Printa a precisão e a precisão média
print('\nPrecisão de cada Fold da Validação Cruzada:' , scores['test_precision'])
print('Precisão Média:' , mean(scores['test_precision']))

# Printa a recall e a recall média
print('\nRecall de cada Fold da Validação Cruzada:' , scores['test_recall'])
print('Recall Média:' , mean(scores['test_recall']))

# Printa a f1 e a f1 média
print('\nf1 de cada Fold da Validação Cruzada:' , scores['test_f1'])
print('f1 Média:' , mean(scores['test_f1']))

