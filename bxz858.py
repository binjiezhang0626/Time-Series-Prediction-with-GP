import sexpdata
import math
import random
import numpy as np
import time
import argparse
from sexpdata import loads, dumps, Symbol
import sys


def preorder(expr, n, x):
    if expr == None or type(expr)!= list:
         return expr

    v_1 = preorder(expr[1], n, x)
    if len(expr) >= 3:
        v_2 = preorder(expr[2], n, x)
    if len(expr) >=5:
        v_3 = preorder(expr[3], n, x)
        v_4 = preorder(expr[4], n, x)
    try:
        if expr[0].value() == 'add':
            result = v_1 + v_2
            return float(result)
        if expr[0].value() == 'sub':
            result = v_1 - v_2
            return float(result)
        if expr[0].value() == 'mul':
            result = v_1*v_2
            return float(result)
        if expr[0].value() == 'div':
            if v_2 == 0:
                return 0
            else:
                result = v_1/v_2
                return float(result)
        if expr[0].value() == 'pow':
            result = math.pow(v_1, v_2)
            return float(result)
        if expr[0].value() == 'sqrt':
            if v_1 <= 0:
                return 0
            else:
                result = math.sqrt(v_1)
                return float(result)
        if expr[0].value() == 'log':
            result = math.log(v_1,2)
            return float(result)
        if expr[0].value() == 'exp':
            result = math.exp(v_1)
            return float(result)
        if expr[0].value() == 'max':
            result = max(v_1, v_2)
            return float(result)
        if expr[0].value() == 'ifleq':
            if v_1 <= v_2:
                result = v_3
                return float(result)
            else:
                result = v_4
                return float(result)
        if expr[0].value() == 'data':
            j = int(abs(math.floor(v_1))) % n
            result = x[j]
            return float(result)
        if expr[0].value() == 'diff':
            k = int(abs(math.floor(v_1))) % n
            l = int(abs(math.floor(v_2))) % n
            result = x[k]-x[l]
            return float(result)
        if expr[0].value() == 'avg':
            k = int(abs(math.floor(v_1))) % n
            l = int(abs(math.floor(v_2))) % n
            t = min(k,l)
            p = max(k,l)
            q = 0
            for i in range(t, p):
                q = q + x[i]
            if k != l:
                return float(q/abs(k-l))
            else:
                return 0
    except ValueError:
        return 0
    except OverflowError:
        return 0
    except ZeroDivisionError:
        return 0
        
def expression(n, x, expr):
    tree = sexpdata.loads(expr)
    answer = preorder(tree, n, x)
    return answer

def readfile(data):
    f = open(data)
    flag = True
    for data in f.readlines():
        nums = data.split("\t")
        if flag:
            nums = [float(i) for i in nums]
            martix = np.array(nums)
            flag = False
        else:
            nums = [float(i) for i in nums] 
            martix = np.c_[martix,nums]
    data1 = martix.transpose()
    f.close()
    return data1

def fitness(expr, n, m, data):

    data = readfile(data)
    vector_x = data[:, 0:n]
    result_y = data[:, -1]
    y_test = []
    for i in range(m):
        x = vector_x[i]       
        y2 = expression(n, x, expr)
        y_test.append(y2)
    error = []
    for i in range(len(result_y)):
        error.append(result_y[i] - y_test[i])
    squaredError = []
    try:
        for val in error:    
            squaredError.append(val * val)
        mse = float(sum(squaredError)) / float(len(squaredError))
        return mse
    except:
        return sys.float_info.max

def initialization(parents_number):
    max_depth = 2
    pop = []
    for i in range(0, parents_number):
        new_individual = dumps(spanning_tree(0, max_depth))
        pop.append([new_individual, 0])
    return pop

def crossover(offspring, p1_index, p2_index):
    p_1 = loads(offspring[p1_index][0])
    p_2 = loads(offspring[p2_index][0])

    position_1 = random.randint(0, node_number(p_1))
    position_2 = random.randint(0, node_number(p_2))

    p1_subtree = find_subtree(p_1, position_1, 0, p_1)[1]
    p2_subtree = find_subtree(p_2, position_2, 0, p_2)[1]

    p_1 = replace(p_1, position_1, p2_subtree, 0)[0]
    p_2 = replace(p_2, position_2, p1_subtree, 0)[0]
    return [[dumps(p_1), 0], [dumps(p_2), 0]]

def mutation(individual):
    tree = loads(individual[0])
    position = random.randint(0, node_number(tree))
    new_branch = spanning_tree(0, 2)
    individual = [dumps(replace(tree, position, new_branch, 0)[0]), 0]
    return individual   

def isTree(tree):
    if type(tree) == list:
        return True
    else:
        return False

def random_oper():
    oper_list = [[Symbol('add'), 2], [Symbol('sub'), 2], [Symbol('mul'), 2], [Symbol('div'), 2],
                     [Symbol('pow'), 2], [Symbol('max'), 2], [Symbol('avg'), 2], [Symbol('diff'), 2],
                     [Symbol('sqrt'), 1], [Symbol('log'), 1], [Symbol('exp'), 1], [Symbol('data'), 1],
                     [Symbol('ifleq'), 4]]
    return oper_list[random.randint(0, len(oper_list) - 1)]

def spanning_tree(depth, max_depth):
    depth = depth + 1
    if depth == 0:
        tree_list = []
        new_node = random_oper()
        tree_list.append(new_node[0])
        for i in range(0, new_node[1]):
            tree_list.append(spanning_tree(depth, max_depth))
    elif depth > max_depth:
        return random.randint(0, 10)
    else:
        tree_list = []
        new_node = random_oper()
        tree_list.append(new_node[0])
        for i in range(0, new_node[1]):
            tree_list.append(spanning_tree(depth, max_depth))
    return tree_list

def node_number(tree):
    node_num = 0
    if isTree(tree):
        node_num = node_num + 1
        for i in range(1, len(tree)):
            node_num = node_num + node_number(tree[i])
    return node_num

def replace(tree, position, new_branch, index):
    if isTree(tree):
        if position == index:
            tree = new_branch
        else:
            for i in range(1, len(tree)):
                if type(tree[i]) == list:
                    index = index + 1
                a = replace(tree[i], position, new_branch, index)
                tree[i] = a[0]
                index = a[1]
    return [tree, index]

def find_subtree(tree, position, index, sub_tree):
    if isTree(tree):
        if position == index:
            sub_tree = tree
        else:
            for i in range(1, len(tree)):
                if type(tree[i]) == list:
                    index = index + 1
                a = find_subtree(tree[i], position, index, sub_tree)
                tree[i] = a[0]
                index = a[2]
                sub_tree = a[1]
    return [tree, sub_tree, index]

def sort_fitness(sort):
    return sort[1]

def genetic_programming(lambda1, n, m, data, time_budget):
    crossover_prob = 0.85
    mutation_prob = 0.2

    start_time = int(time.time())

    pop = initialization(lambda1)
    for i in range(0, lambda1):
        pop[i][1] = fitness(pop[i][0], n, m, data)

    itr = 0
    while ((int(time.time()) - start_time) <= time_budget):
        itr = itr + 1
        parents = pop[0:lambda1]
        offspring = parents[:]

        for i in range(0, int(lambda1 / 2)):
            if random.random() < crossover_prob:
                p_1_index = int(random.random() * lambda1)
                p_2_index = int(random.random() * lambda1)
                crossovered = crossover(offspring, p_1_index, p_2_index)
                offspring[p_1_index] = crossovered[0]
                offspring[p_2_index] = crossovered[1]

        for i in range(0, lambda1):
            if random.random() < mutation_prob:
                offspring[i] = mutation(offspring[i])

        for i in range(0, lambda1):
            offspring[i][1] = fitness(offspring[i][0], n, m, data)

        pop = parents + offspring
        pop.sort(key=sort_fitness)
        
    return pop[0][0]

parser = argparse.ArgumentParser()

parser.add_argument("-question", help="should given question 1,2 or 3", type=int)
#the dimension of input vector
parser.add_argument("-n", help="should input an vector size", type=int)
#the size of the trainning data
parser.add_argument("-m", help="should input an data size", type=int)
#input vector it is string not list
parser.add_argument("-x", help="should input a vector", type=str)
#should input an expression
parser.add_argument("-expr", help="should input an expression", type=str)
#input the file name
parser.add_argument("-data", help="should input an file name", type=str)
#input pop size
parser.add_argument("-lambdainput", help="should input pop size", type=int)
#the time budget
parser.add_argument("-time_budget", help="should input the time budget", type=float)

args = parser.parse_args()

if args.question==1 :
    x=args.x
    x1 = x.split(" ")
    x = [float(x) for x in x1]
    result_1=expression(args.n, x, args.expr)
    print(result_1)

if args.question==2:
    result_2=fitness(args.expr, args.n, args.m, args.data)
    print (result_2) 

if args.question==3:
    result_3=genetic_programming(args.lambdainput, args.n, args.m, args.data, args.time_budget)    
    print (result_3)