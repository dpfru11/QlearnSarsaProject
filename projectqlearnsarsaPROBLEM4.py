import numpy as np
from numpy.lib.function_base import average
from numpy.lib.twodim_base import triu_indices_from
import matplotlib.pyplot as plt
import random

MAX_T = 250
N_EPISODES = 500
EPS = 0.1
#varyingEPS = 0.1
twoGoal = True
ALPHA = 0.7
ALPHA_SAR = 0.7

rows = 7
cols = 13
env = {}

#3D Value: row, column, and specified action
q_table = np.zeros((rows, cols, 4))
#q_table = np.full((rows, cols, 4), fill_value=-1000)

rewards = np.full((rows, cols), -1)
rewards[0, cols - 1] = 100
rewards[rows - 1, cols - 1] = 100
for i in range(5,12):
    rewards[3, i] = -100

rewardsOverRuns = []

startingLoc = (3,0)

#Print rewards table
for row in rewards:
    print(row)

#Action space by row, range (NOT (x, y))
up=(-1,0)
down=(1,0)
left=(0,-1)
right=(0,1)
actionList = [up, down, left, right]
actionStrings = ["U", "D", "L", "R"]


"""Check if Terminal State"""
def isEnd(state, step):
    if step == MAX_T:
        return True
    elif rewards[state[0], state[1]] == -1 or rewards[state[0], state[1]] == -100:
        return False
    else:
        return True

"""Simple getter for starting location"""
def getStart():
    return startingLoc

"""Returns the following state after executing an action"""
def getSucc(currentState, action):
    zipped = zip(currentState, action)
    mapped = map(sum, zipped)
    return tuple(mapped) #New State/Location

"""Finds the next action using the epsilon-greedy method"""
def getAction(currentState, varyEPS):
    index = 0
    if random.random() < 1 - varyEPS:
        index = np.argmax(q_table[currentState[0], currentState[1]])
    else:
        index = random.randint(0,3)
        
    return actionList[index]

"""
Takes the shortest path on the grid BASED ON THE LAST USED METHOD (Q/SARSA).
A few bugs:
1. Does not work if algorithm created a policy cycle in the graph (Just keep displaying the same cell until you run out of steps)
2. If policy leads you out of the matrix, crashes the script. (Simple boundary checking can fix this, will bother with later since crashes are not frequent)
Not for grade, but for myself ¯\_(ツ)_/¯
"""
def shortestPath(startState):
    step = 1
    currentState = startState
    shortestPathList = []
    shortestPathList.append(currentState)
    step += 1
    while not isEnd(currentState, step):
        index = np.argmax(q_table[currentState[0], currentState[1]])
        nextAction = actionList[index]
        currentState = getSucc(currentState, nextAction)
        shortestPathList.append(currentState)
        step += 1
    return shortestPathList

"""OFF-POLICY: Q-Value Equation: Q(S,A) <- Q(S,A) + ALPHA * [Reward + GAM * Q(S', A') - Q(S,A)]"""
def Qlearn():
    runReward = []
    varyEPS = EPS
    for episode in range(N_EPISODES):
        step = 0
        currentState = getStart()
        totalReward = 0
        while not isEnd(currentState, step):
            
            currentAction = getAction(currentState, varyEPS)
            
            newState = getSucc(currentState,currentAction)

            #Check Boundaries, waste a step if out of boundary
            if newState[1] >= cols or newState[1] <= 0:
                newState = currentState
            elif newState[0] >= rows or newState[0] <= 0:
                newState = currentState
            
            reward = rewards[newState[0], newState[1]]
            if reward != 100:
                totalReward += reward
            
            oldq = q_table[currentState[0], currentState[1], actionList.index(currentAction)]

            #Calculate Q-Value and assign to the Q-Table
            newq = oldq + ALPHA * (reward + np.max(q_table[newState[0], newState[1]]) - oldq)
            q_table[currentState[0], currentState[1], actionList.index(currentAction)] = newq
            
            #Start over (don't end the episode)
            if rewards[newState[0], newState[1]] == -100:
                currentState = getStart()
            else:
                currentState = newState
            step += 1
        runReward.append(totalReward)
        #if varyEPS != 0:
         #   varyEPS = varyEPS - 0.1/500
    print("End Training")
    #print(runReward)
    return runReward

"""ON-POLICY: Q-Value Equation: Q(S,A) <- Q(S,A) + ALPHA * [Reward + GAM * Q(S', A') - Q(S,A)]"""
def SARSA():
    runReward = []
    varyEPS = EPS
    for episode in range(N_EPISODES):
        step = 0
        currentState = getStart()
        currentAction = getAction(currentState, varyEPS)
        totalReward = 0
        while not isEnd(currentState, step):
            
            newState = getSucc(currentState,currentAction)
            
            #Boundary Checking
            if newState[1] >= cols or newState[1] < 0:
                newState = currentState
            elif newState[0] >= rows or newState[0] < 0:
                newState = currentState
            
            newAction = getAction(newState, varyEPS)

            reward = rewards[newState[0], newState[1]]
            if reward != 100:
                totalReward += reward
            
            #Calculate Q-Value and assign to the Q-Table
            oldq = q_table[currentState[0], currentState[1], actionList.index(currentAction)]
            newq = oldq + ALPHA_SAR * (reward + q_table[newState[0], newState[1], actionList.index(newAction)] - oldq)
            q_table[currentState[0], currentState[1], actionList.index(currentAction)] = newq

            #Jump to start
            if rewards[newState[0], newState[1]] == -100:
                currentState = getStart()
                currentAction = getAction(currentState, varyEPS)
            else:
                currentState = newState
                currentAction = newAction
            
            step += 1
        runReward.append(totalReward)
        #if varyEPS != 0:
         #   varyEPS = varyEPS - 0.1/500
    print("End Training")
    #print(runReward)
    return runReward

#Run QLearn Algorithm 10 times
for i in range(0, 10):
    q_table = np.zeros((rows, cols, 4))
    rewardsOverRuns.append(Qlearn())

print("Shortest path QL:\n")
#print(shortestPath(getStart()))
print("averaging")
avgsQL = [float(sum(col))/len(col) for col in zip(*rewardsOverRuns)]

#Visual display of optimal policies for each cell
print("Policy")
for row in range(0,rows):
    for col in range(0,cols):
        if col == cols - 1:
            print("(" + str(row) + ", "+ str(col) + ")" + "{" + str(actionStrings[np.argmax(q_table[row, col])]) + "}")
        else:   
            print("(" + str(row) + ", "+ str(col) + ")" + "{" + str(actionStrings[np.argmax(q_table[row, col])]) + "}", end='')

rewardsOverRuns = []
#Run SARSA Algorithm 10 times
for i in range(0, 10):
    q_table = np.zeros((rows, cols, 4))
    rewardsOverRuns.append(SARSA())
print("averaging")
avgsSA = [float(sum(col))/len(col) for col in zip(*rewardsOverRuns)]

xaxis = []
for i in range(0, 500): xaxis.append(i)


#Visual display of optimal policies for each cell
print("Policy")
for row in range(0,rows):
    for col in range(0,cols):
        if col == cols - 1:
            print("(" + str(row) + ", "+ str(col) + ")" + "{" + str(actionStrings[np.argmax(q_table[row, col])]) + "}")
        else:   
            print("(" + str(row) + ", "+ str(col) + ")" + "{" + str(actionStrings[np.argmax(q_table[row, col])]) + "}", end='')
print("Shortest path SARSA:\n")
#print(shortestPath(getStart()))

#Plotting average reward obtained from the episodes of each algorithm
plt.plot(xaxis, avgsQL, label = "Qlearning Average Rewards")
plt.plot(xaxis, avgsSA, label = "SARSA Average Rewards")
plt.xlabel("Number of Episodes")
plt.ylabel("Average Reward")
plt.legend()
plt.show()
