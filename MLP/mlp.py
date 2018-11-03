"""
 In this file was implemented a multilayer perceptron(MLP) with multiples hidden layers. You can specify the numbers of hidden layers
 as well as the numbers of neurons inside each layers(input, hiddens and output layers)
"""

import numpy as np

"""
Função de ativação sigmóide
"""
def f_net(net):
    return (1/(1+np.exp(-net)))
"""
Derivada da função de ativação
"""
def df_dnet(f_net):
    return f_net*(1-f_net)

"""
Cria as matrizes que irão representar os pesos de ambas as camadas utilizando valores aleatórios entre -0.5 e 0.5
"""
def mlpArchitecture(lenghts):
    weights=[]
    for i in range(1, len(lenghts)):
        weights.append(np.random.uniform(low=-0.5, high=0.5, size=(lenghts[i], lenghts[i-1]+1)))
    weights.append([])
    return weights

"""
Realiza a etapa de forward com os pesos que estão representados em "model" sobre os dados de entrada representados em "inPut".
Retorna todos os fnets e derivadas de fnets de todos as camadas e neurônios.
"""
def forward(model, inPut):
    fNet=[0]*len(model)
    dfdNet=[0]*len(model)
    fNet[0]=inPut
    dfdNet[0]=df_dnet(fNet[0])
    for i in range(1, len(model)):
        net=model[i-1].dot(np.vstack((fNet[i-1],[1])))
        fNet[i]=f_net(net)
        dfdNet[i]=df_dnet(fNet[i])
    return {"fNet":fNet, "dfdNet": dfdNet}

"""
Realiza o treinamento da MLP e retorna o modelo treinado.
Parametros:
    inData: array contendo nas linhas as instancias e nas colunas os atributos dos dados de entrada
    outData: array contendo nas linhas as instancias e nas colunas os atributos dos dados de saída
    layers: lista representandp o número de camadas e o número de neurônios de cada uma delas
    eta: parametro momento que implica na velocidade da conversão
    threshold: indica a diferença minima entre erros consecutivos na iterações para que o algoritmo pare.
    maxIters: indica o número máximo de iterações
"""
def mlp(inData, outData, layers=[2,2,1], eta=0.1, threshold=1e-8, maxIters=5000):
    model=mlpArchitecture(layers)
    numLayers=len(layers)
    counter=0
    prevError=0
    diffError=2*threshold

    # Executa enquanto a diferença entre erros consecutivos não seja menor que o mínimo ou
    # até que se alcance o número de iterações máxima
    while(diffError > threshold and counter < maxIters):
        erroSomado=0
        counter+=1

        # realiza uma randomização na ordem com que as instâncias vão ser utilizadas para treino, de modo a não
        # enviezar a convergência
        shuffle=np.random.permutation(range(0, inData.shape[0]))

        # executa para todas as instâncias
        for i in range(0, inData.shape[0]):
            # realiza o forward
            fow=forward(model, inData[[shuffle[i]]].T)
            erro=outData[[shuffle[i]]] - fow['fNet'][numLayers-1].T
            erroSomado+=np.power(erro,2).sum()
            model[numLayers-1]=np.hstack((2*erro, [[1]]))
            delta=np.array([[1]])
            
            # realiza o cálculo de todas as derivadas e atualiza os pesos
            for j in range(numLayers-1, 0, -1):
                colSize=model[j].shape[1]
                delta=delta.T.dot(model[j][:,:colSize-1]).T * fow["dfdNet"][j]
                dedw=delta.dot(np.vstack((fow["fNet"][j-1],[1])).T)
                model[j-1] = model[j-1] + eta*dedw
                #print(j)
        #print(model)
        erroSomado/=inData.shape[0]
        diffError=abs(prevError-erroSomado)
        prevError=erroSomado
        #print(diffError)
    return model
