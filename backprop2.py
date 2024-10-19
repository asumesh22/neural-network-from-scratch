import sys; args = sys.argv[1:]
import math
import random

def main():
    function = f
    alpha = 1.5 # change
    epoch_count = 1000 # change

    inputs, outputs, nodes_per_layer, weights = parse(args)

    bestError = 10000
    bestWeights = []
    
    print(f'Layer counts: {nodes_per_layer}')
    for epoch in range(epoch_count):
        E = 0

        for idx, inp in enumerate(inputs):
            nodes = [[0]*nodes_per_layer[i] for i in range(len(nodes_per_layer))]
            nodes[0] = inp

            nodes = forward_propagation(nodes, weights, function)
            E += getError(nodes, outputs[idx])
            weights = back_propagation(nodes, weights, outputs[idx], dfdx, alpha)

        if E < bestError:
            bestError = E
            bestWeights = weights
            if E < 200: alpha = 0.1
            if E < 100: alpha = 0.075
            if E < 50: alpha = 0.05
            if E < 40: alpha = 0.04
            if E < 30: alpha = 0.025
            if E < 15: alpha = 0.0125
        if epoch % 10 == 0:
            prnt(bestWeights)
            print('----------------------------------', bestError)
        
    # print('----------------------------------', bestError)
            
def prnt(weights):
    for layerNum, layer in enumerate(weights):
        if layerNum == len(weights)-1: 
            print(' '.join([str(y) for x in layer for y in x]))
        else:
            for l in range(len(layer[0])):
                for node in layer:
                    if node[l]: print(node[l], end=" ")
            print()
    print()

def parse(args):
    inputs = args[0]
    sym = ''
    if '>=' in inputs:
        inputs = inputs.split('>=')
        sym = '>='
    elif '<=' in inputs:
        inputs = inputs.split('<=')
        sym = '<='
    elif '>' in inputs:
        inputs = inputs.split('>')
        sym = '>'
    else:
        inputs = inputs.split('<')
        sym = '<'
    
    Rsquared = float(inputs[-1])
    inputs = []; outputs = []

    input_file = []

    numInputs = 10000
    o = False
    for i in range(numInputs):
        # random inputs in range -1.5 to 1.5
        # x = random.random()*3-1.5
        # y = random.random()*3-1.5
        x = (random.random() - 0.5) * 3
        y = (random.random() - 0.5) * 3

        if o:
            # needs to be outside circle
            while x*x + y*y < Rsquared:
                x = (random.random() - 0.5) * 3
                y = (random.random() - 0.5) * 3
        else:
            # needs to be inside circle
            while x*x + y*y > Rsquared:
                x = (random.random() - 0.5) * 3
                y = (random.random() - 0.5) * 3
        o = not o

        if sym == '>=':
            if x*x + y*y >= Rsquared:
                input_file.append(f"{x} {y} => 1")
            else:
                input_file.append(f"{x} {y} => 0")
        if sym == '<=':
            if x*x + y*y <= Rsquared:
                input_file.append(f"{x} {y} => 1")
            else:
                input_file.append(f"{x} {y} => 0")
        if sym == '>':
            if x*x + y*y > Rsquared:
                input_file.append(f"{x} {y} => 1")
            else:
                input_file.append(f"{x} {y} => 0")
        if sym == '<':
            if x*x + y*y < Rsquared:
                input_file.append(f"{x} {y} => 1")
            else:
                input_file.append(f"{x} {y} => 0")

    inputs = []; outputs = []

    for row in input_file:
        s = [i.strip() for i in row.split("=>")]
        inputs.append([float(i) for i in (s[0] + " 1").split(" ")]) # bias
        outputs.append([float(i) for i in s[1].split(" ")])

    # [3, 3, 2, 2, 1, 1] goof ct 19162
    # [3, 12, 6, 1, 1] goof ct 12879
    # [3, 6, 3, 2, 1, 1] goof ct 8274
    nodes_per_layer = [3, 6, 3, 2, 1, 1]
    weights = [[[random.random()*2-1 for k in range(nodes_per_layer[i+1])] for j in range(nodes_per_layer[i])] for i in range(len(nodes_per_layer)-1)]
    weights[-1] = [n[:1] for n in weights[-1]]

    return inputs, outputs, nodes_per_layer, weights

def back_propagation(nodes, weights, output, fprime, alpha):
    G = [[[0]*len(j) for j in i] for i in weights]
    E = [[0]*len(nodes[i]) for i in range(len(nodes))]

    for layer in range(len(nodes)-2, -1, -1):
        for i in range(len(nodes[layer])):
            if layer != len(nodes)-2:
                error = 0
                for j in range(len(E[layer+1])):
                    grad = nodes[layer][i] * E[layer+1][j]
                    G[layer][i][j] = grad * alpha
                    error += E[layer+1][j] * weights[layer][i][j]
                E[layer][i] = fprime(nodes[layer][i]) * error

            else:
                grad = nodes[layer][i] * (output[i]-nodes[layer+1][i])
                E[layer][i] = fprime(nodes[layer][i]) * weights[layer][i][0] * (output[i]-nodes[layer+1][i])
                G[layer][i][0] = grad * alpha
                
    for layer in range(len(G)):
        for i in range(len(G[layer])):
            for j in range(len(G[layer][i])):
                weights[layer][i][j] += G[layer][i][j]

    #print(weights)
    #exit()

    return weights

def forward_propagation(nodes, weights, function):
    for layer in range(len(nodes)-1):
        this_layer_nodes = nodes[layer]
        this_layer_weights = weights[layer]
        next_layer_nodes = [0 for i in nodes[layer+1]]

        if layer == len(nodes)-2:
            vec = [n[-1] for n in this_layer_weights]
            next_layer_nodes = hadamard_product(vec, this_layer_nodes)
        else:
            for i in range(len(next_layer_nodes)):
                vec = [n[i] for n in this_layer_weights]
                next_layer_nodes[i] = function(dot_product(this_layer_nodes, vec))
            
        nodes[layer+1] = next_layer_nodes

    # print('Feed forawrd', nodes)

    return nodes

def hadamard_product(a, b):
    return [a[i]*b[i] for i in range(len(a))]

def dot_product(a, b):
    return sum(hadamard_product(a, b))

def f(x):
    return 1/(1+math.exp(-x))

def dfdx(x):
    return (1-x)*x

def getError(nodes, outputs):
    return 0.5 * sum([(outputs[i]-nodes[-1][i])**2 for i in range(len(outputs))])

if __name__ == '__main__': main()
# Aaryan Sumesh, Pd. 2, Class of 2025