import sys; args = sys.argv[1:]
import math
import random

def main():
    function = f
    alpha = 0.1 # change
    epoch_count = 100000 # change

    inputs, outputs, nodes_per_layer, weights = parse(args)

    print('inputs', inputs)
    print('weights', weights)

    bestError = 1000
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
            # print('----------------------------------', bestError)

        '''z = True
        if z:
            for x in weights:
                if z:
                    for y in x:
                        if z:
                            for a in y:
                                if a < 30:
                                    z = False

        if not z:
            print('----------------- resetting weights -----------------')
            inputs, outputs, nodes_per_layer, weights = parse(args)'''


    prnt(bestWeights)
    print('----------------------------------', bestError)
            
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
    input_file = open(args[0]).read().splitlines()
    input_file.sort()
    inputs = []; outputs = []

    for row in input_file:
        s = [i.strip() for i in row.split("=>")]
        inputs.append([float(i) for i in (s[0] + " 1").split(" ")]) # bias
        outputs.append([float(i) for i in s[1].split(" ")])

    nodes_per_layer = [len(inputs[0]), 3, len(outputs[0]), len(outputs[0])]
    weights = [[[.5 for k in range(nodes_per_layer[i+1])] for j in range(nodes_per_layer[i])] for i in range(len(nodes_per_layer)-1)]
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
    return [a[i]*b[i] for i in range(min(len(a), len(b)))]

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