import math
import random
import csv
import time

# read csv file
def importData(filePath='mnist.csv'):
    mnist = []
    with open(filePath) as file:
        csvFile = csv.reader(file)
        for lineNum, line in enumerate(csvFile):
            if lineNum == 0: continue # don't read headings
            testCase = []
            for n in line:
                testCase.append(int(n)) # convert str to int
            mnist.append(testCase) # list of lists
    return mnist

# split data into training and testing
def splitData(data, fraction=0.9):
    random.shuffle(data) # randomize
    newData = []
    for row in data:
        numberLine = [0]*10
        numberLine[row[0]] = 1
        newData.append([numberLine] + row[1:])
    data = newData
    n = int(len(data) * fraction) # take first fraction of data
    trainingData = data[:n]
    testingData = data[n+1:]
    return trainingData, testingData

# train neural network
def trainNetwork(layerCounts, weights, trainingData, numEpochs=10, updateEvery = 10, alpha=0.03):
    bestWeights = weights
    minError = float('inf')
    prevError = -1
    foundNewBest = False
    for epochNumber in range(numEpochs):
        error = 0
        for idx, data in enumerate(trainingData):
            nodes = [[0]*n for n in layerCounts]
            if idx and idx % 1000 == 0: print(f'Epoch {epochNumber} - Testing input test case {idx} of {len(trainingData)} - current error {error/idx}')
            inp = data[1:]
            out = data[0]
            nodes[0] = inp
            nodes = feedForward(weights, nodes)
            weights = backProp(nodes, weights, out, alpha)
            error += networkError(nodes, out)
        error = error / len(trainingData)
        if error < minError:
            minError = error
            bestWeights = weights
            foundNewBest = True
        if epochNumber % updateEvery == 0:
            if foundNewBest:
                with open('output.txt', 'w') as file:
                    weightStr = weightsToString(bestWeights)
                    file.write(weightStr)
                    print('Update: Wrote weights to file')
            print(f'Update: training on epoch number {epochNumber} with minimum error {minError} per test case')
            foundNewBest = False
        if abs(prevError-error) < 0.001:
            print('Update: reached local min, resetting weights')
            weights = generateWeights(layerCounts, factor = 100)
        if error < 0.001:
            return bestWeights
        
        prevError = error
    return bestWeights

# error of network
def networkError(nodes, output):
    e = 0.5 * sum((output[i] - nodes[-1][i])**2 for i in range(len(output)))
    return e

# weights to string
def weightsToString(weights):
    st = ""
    for layer in weights:
        st += "[" + ", ".join([str(x) for x in layer]) + "]" + "\n"
    return st

# feed forward
def feedForward(weights, nodes):
    for layer in range(len(nodes) - 1):
        if layer == len(nodes) - 2: # last layer
            nodes[layer+1] = hadamardProduct([n[-1] for n in weights[layer]], nodes[layer]) # only scale (no butterfly)
        else:
            for i in range(len(nodes[layer+1])): # not last layer
                nodes[layer+1][i] = f(dotProduct(nodes[layer], [n[i] for n in weights[layer]])) # butterfly network
    return nodes

# backprop
def backProp(nodes, weights, output, alpha):
    gradients = [[[0]*len(j) for j in i] for i in weights] # gradients
    errors = [[0]*len(nodes[i]) for i in range(len(nodes))] # errors

    for layer in range(len(nodes)-2, -1, -1):
        for i in range(len(nodes[layer])):
            if layer != len(nodes) - 2: # not first backprop layer
                error = 0
                for j in range(len(errors[layer+1])):
                    gradient = nodes[layer][i] * errors[layer+1][j]
                    gradients[layer][i][j] = gradient * alpha
                    error += errors[layer+1][j] * weights[layer][i][j]
                errors[layer][i] = deriv(nodes[layer][i]) * error
            else:
                gradient = nodes[layer][i] * (output[i] - nodes[layer+1][i])
                errors[layer][i] = deriv(nodes[layer][i]) * weights[layer][i][0] * (output[i] - nodes[layer+1][i])
                gradients[layer][i][0] = gradient * alpha

    for layer in range(len(gradients)):
        for i in range(len(gradients[layer])):
            for j in range(len(gradients[layer][i])):
                weights[layer][i][j] += gradients[layer][i][j]

    return weights

# random seeding of weights
def generateWeights(layerCounts, factor = 1):
    weights = [[[(random.random()*2-1)/factor for k in range(layerCounts[i+1])] for j in range(layerCounts[i])] for i in range(len(layerCounts)-1)]
    weights[-1] = [n[:1] for n in weights[-1]]
    return weights

# function
def f(x):
    return x # linear
    return 1/(1+math.exp(-x)) # logistic
    # return max(0, x) # relu

# derivative of logistic function
def deriv(x):
    return 1 # linear
    return x * (1-x) # logistic
    return 0 if x < 0 else 1 # relu

# hadamard product
def hadamardProduct(a, b):
    return [a[i]*b[i] for i in range(len(a))]

# dot product
def dotProduct(a, b):
    return sum(hadamardProduct(a, b))

# main
def main():
    mnist = importData() # import data
    trainingData, testingData = splitData(mnist, fraction=0.9) # partition data
    layerCounts = [785, 32, 10, 10]
    weights = generateWeights(layerCounts, factor = 100)
    t1 = time.time()
    weights = trainNetwork(layerCounts, weights, trainingData, updateEvery=10, alpha=0.0003)
    elapsedTime = time.time()-t1
    print(f'Network Training took {elapsedTime}s')

    # determine error
    error = 0
    for idx, data in enumerate(testingData):
        if idx and idx%1000 == 0: print(f'Calculating error, test case {idx}: error is currently {error/idx} per test case')
        nodes = [[0]*n for n in layerCounts]
        inp = data[1:]
        out = data[0]
        nodes[0] = inp
        nodes = feedForward(weights, nodes)
        error += networkError(nodes, out)

    print(f'Final network has error {error}, with {error/len(testingData)} error per test case')

if __name__ == '__main__': main()
# Aaryan Sumesh