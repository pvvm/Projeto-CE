from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions

#def plot_fronteiras(n_vizinhos):
#    knn = KNeighborsClassifier(n_neighbors=n_vizinhos)
#    knn.fit(X, y)
#    plt.figure(figsize=(8,5))
#    plot_decision_regions(X,y,clf=knn,legend=2)
#    plt.xlabel('alcohol')
#    plt.ylabel('malic_acid')
#    plt.title('Fronteiras de Complexidade - KNN')
#    plt.show()

# Carrego o dataset
cancer = load_breast_cancer()

# Cria o dataframe
df_cancer = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
#Cria coluna com valores de target
df_cancer['class'] = cancer.target

#print(df_cancer)
#print(df_cancer.head().T)
#print(df_cancer.info())
#print(df_cancer['class'].value_counts())

# Separa os dados de treino e teste e define a quantidade de teste
X_train, X_test, y_train, y_test = train_test_split(df_cancer.drop('class', axis=1), df_cancer['class'], test_size=0.3)

# Define o numero de vizinhos e metrica
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# Aplica o treino no KNN
knn.fit(X_train, y_train)
# Executa o KNN no teste
resultado = knn.predict(X_test)
#print(resultado)

print(pd.crosstab(y_test, resultado, rownames=['Real'], colnames=['Predito'], margins=True))
print(metrics.classification_report(y_test, resultado, target_names=cancer.target_names))

# OTIMIZANDO O K #

# Define lista de valores
#k_list = list(range(1,31))
# Coloca valores no dicionario
#parametros = dict(n_neighbors=k_list)
# Instancia o objeto GridSearch, com modelo, numero de vizinhos, numero de dobreas e a metrica de avaliacao
#grid = GridSearchCV(knn, parametros, cv=5, scoring="accuracy")
#grid.fit(df_cancer.drop('class', axis=1), df_cancer['class'])
#print("Melhores parametros {} com o valor de acuracia {} ".format(grid.best_params_,grid.best_score_))
#plt.figure(figsize=(10,6))
#grid.score esta errado, nao funciona esse plt.plot
#plt.plot(k_list,grid.score,color='red',linestyle='dashed',marker='o')
#plt.show()

# VISUALIZAR FRONTEIRAS #

#X = cancer.data[:,[0,2]]
#y = cancer.target

#plot_fronteiras(3)