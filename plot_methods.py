import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
import os

home_path = os.getcwd()
plot_path = home_path + '/plots/'
log_path = home_path + '/logs/'

def readLossHist(filename): 
    f = open(log_path + 'loss_hiss/' + filename + '.txt', "r")
    lines = f.readlines()
    data = [ast.literal_eval(line) for line in lines]
    f.close()
    return data[0] 

def plotLoss(rnn, lossHist, num_iterations):

    plotTitle = "Smooth loss for {rnnType}, after {num_iterations} iterations".format(rnnType = rnn.type, num_iterations = num_iterations)

    # plot results
    steps = [step * 10 for step in range(len(lossHist))]
    plt.plot(steps, lossHist, 'r', linewidth=1.5, alpha=1.0, label='Loss')
    plt.xlim(0, steps[-1])
    plt.xlabel('Iterations')
    plt.ylabel('Loss', rotation=0, labelpad=20)
    plt.title(plotTitle)
    plt.savefig(plot_path + str(rnn.type) + '_' + str(num_iterations) + '_iter_loss.png', dpi=200)
    plt.clf()

def plotLossBare(rnn_type, num_iterations, lossHist):

    plotTitle = "Smooth loss for {rnnType}, after {num_iterations} iterations".format(rnnType = rnn_type, num_iterations = num_iterations)
    # plot results
    steps = [step * 10 for step in range(len(lossHist))]
    plt.plot(steps, lossHist, 'r', linewidth=1.5, alpha=1.0, label='Loss')
    plt.xlim(0, steps[-1])
    plt.xlabel('Iterations')
    plt.ylabel('Loss', rotation=0, labelpad=20)
    plt.title(plotTitle)
    plt.savefig(plot_path + str(rnn_type) + '_' + str(num_iterations) + '_iter_loss.png', dpi=200)
    plt.clf()

def multiPlotLoss(rnn_list, num_iterations, lossHist_list, test_set):

    plotTitle = "Smooth loss on {test_set} set".format(test_set = test_set)

    line_colors = ['r','g','b','c','m','y']

    # plot results
    steps = [step * 10 for step in range(len(lossHist_list[0]))]
    for idx, rnn in enumerate(rnn_list): 
        plt.plot(steps, lossHist_list[idx], line_colors[idx], linewidth=1.5, alpha=1.0, label=rnn.type)
    plt.legend()
    plt.xlim(0, steps[-1])
    plt.xlabel('Iterations')
    plt.ylabel('Loss', rotation=0, labelpad=20)
    plt.title(plotTitle)
    plt.savefig(plot_path + 'multi_' + str(test_set) + '_' + str(num_iterations) + '_iter_loss.png', dpi=200)
    plt.clf()

def multiPlotLossBare(rnn_type_list, num_iterations, lossHist_list):

    plotTitle = "Smooth loss after {num_iterations} iterations".format(num_iterations = num_iterations)

    line_colors = ['r','g','b','c','m','y']

    # plot results
    steps = [step * 10 for step in range(len(lossHist_list[0]))]
    for idx, rnn in enumerate(rnn_type_list): 
        plt.plot(steps, lossHist_list[idx], line_colors[idx], linewidth=1.5, alpha=1.0, label=rnn)
    plt.legend()
    plt.xlim(0, steps[-1])
    plt.xlabel('Iterations')
    plt.ylabel('Loss', rotation=0, labelpad=20)
    plt.title(plotTitle)
    plt.savefig(plot_path + 'multi_' + str(num_iterations) + '_iter_loss.png', dpi=200)
    plt.clf()

def multiPlotLossHiddenLayer(rnn_type, num_iterations, lossHist_list, test_set):

    plotTitle = "Smooth loss on {test_set} set".format(test_set = test_set)

    # line_colors = ['#ffc9bb','#ff8164','#ff4122','#c61a09','#b60503']
    line_colors = ['r','g','b','c','m','y']
    m_labels = ['m = 10', 'm = 50', 'm = 100', 'm = 150', 'm = 200']

    # plot results
    steps = [step * 10 for step in range(len(lossHist_list[0]))]
    for idx, m in enumerate(m_labels): 
        plt.plot(steps, lossHist_list[idx], line_colors[idx], linewidth=1.5, alpha=1.0, label=m)
    plt.legend()
    plt.xlim(0, steps[-1])
    plt.xlabel('Iterations')
    plt.ylabel('Loss', rotation=0, labelpad=20)
    plt.title(plotTitle)
    plt.savefig(plot_path + str(rnn_type) + '_m_' + str(test_set) + '_'+ str(num_iterations) + '_iter_loss.png', dpi=200)
    plt.clf()

def paramSearchHeatmap(rnn, num_iterations, paramName1, paramList1, paramName2, paramList2, bestLoss_matrix):

    plotTitle = "Hyperparameter evaluation with gridsearch for {rnnType}".format(rnnType = rnn.type)

    # plot results
    f, ax = plt.subplots(figsize=(8, 6))
    vmin = np.min(bestLoss_matrix) - 0.3*(np.max(bestLoss_matrix)- np.min(bestLoss_matrix))
    vmax = np.max(bestLoss_matrix) + 0.3*(np.max(bestLoss_matrix)- np.min(bestLoss_matrix))
    sns.heatmap(bestLoss_matrix, annot=True, xticklabels=paramList1, yticklabels=paramList2, fmt=".2f", cmap = sns.cm.rocket_r, vmin=vmin, vmax=vmax)
    plt.title(plotTitle)
    plt.xlabel(paramName1)
    plt.ylabel(paramName2)
    plt.savefig(plot_path + str(rnn.type) + '_' + str(num_iterations) + '_heatmap.png', dpi=200)
    plt.clf()

class testRNN:
    type = "TestRNN"

if __name__ == "__main__":
    pass
    #data = [[1,2],[3,4]]
    #df = pd.DataFrame(data, columns=['a','b'])
    
    testRNN = testRNN()
    paramSearchHeatmap(testRNN, 100, 'eta', ['0.1','0.2'], 'sigma', ['0.3','0.4'], [[90,95],[100,105]])
    #data1 = readLossHist('VanillaRNN_500_201937')
    #data2 = readLossHist('VanillaRNN_500_202254')
    #multiPlotLossBare(['VanillaRNN1','VanillaRNN2'], '500', [data1, data2])