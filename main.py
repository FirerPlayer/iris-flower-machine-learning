#IMPORTS
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

#CARREGAMENTO DOS DATAFRAMES
iris = pd.read_csv('iris.csv')

#PREPARAÇÃO DOS DADOS
codificador = LabelEncoder()                      #cria um codificar usando a classe sklearn.preprocessing com o metodo Label Encoder. Nesse caso, ele passa as especies de iris para numeros.
X = iris.drop(['species'], axis=1)                #entrada de dados das dimensões
y = codificador.fit_transform(iris['species'])    #saida de dados transformada, mostra a classificação das iris. fit_transform muda as strings para numeros
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)           #divide o dataframe de modo aleatorio na seguinte proporção: 30% para testes e 70% para treinar

#TREINAMENTO DO MODELO E ACURÁCIA
clf = tree.DecisionTreeClassifier()         #inicia o classificador, nesse caso escolhi uma árvore de decisões
clf.fit(X_train, y_train)                   #treina a maquina de acordo com os dados de treino

resultadoTreino = clf.predict(X_train)      #Executa as previsões de treino e de teste com seus respectivos dados de entrada.
resultadoTeste = clf.predict(X_test)

acuraciaTreino, acuraciaTeste = accuracy_score(y_train, resultadoTreino), accuracy_score(y_test, resultadoTeste)        #Mede a acurácia por meio da função score. Acurácia = (Numero de acertos)/len(data)

#PRINTA NA TELA OS DATAFRAMES ORIGINAIS COM AS PREDIÇÕES
print('                             TREINO')
X_train = X_train.assign(ORIGINAL=codificador.inverse_transform(y_train), PREDICAO=codificador.inverse_transform(resultadoTreino))      #uso do codificador para passar de numero para a especie da iris
print(X_train)

print('                             TESTE')
X_test = X_test.assign(ORIGINAL=codificador.inverse_transform(y_test), PREDICAO=codificador.inverse_transform(resultadoTeste))          #uso do codificador para passar de numero para a especie da iris
print(X_test)

print(f'\nAcurácia do treino= {acuraciaTreino:.2f}')
print(f'Acurácia do teste= {acuraciaTeste:.2f}')

'''
Observações: 
#aleterar o random_state em train_test_split modifica o espaço de dados utilizados

A acuracia de 1 não é a esperada com problemas mais reais e complexos. Nesse caso, esse valor ocorreu provavelmente pela complexidade baixa do problema
e a quantidade de dados que é pequena, além de que utilizamos o mesmo dataframe para treino e teste da maquina.
Escolhi utilizar uma arvore de decisões para modelar pois ela apresenta, dentre outras coisas, um bom desempenho em classificações, visto que acontece sucessivas "reduções" de dados 
e que parametros não lineares costumam não afetar sua performance.
'''
