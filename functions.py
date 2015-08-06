#!/usr/bin/python
# Lucius Bynum 2015
################################# DATA FORMATTING ###################################

import random
import svmlight
import time
import datetime

def sparseToList(filename):
    """
    converts to python svm-light format from sparse LIBSVM/svm-light format  
    
    inputs: filename, a string with the name of the file (include '.txt'/ file must be in the path)
    outputs: the training data in python svm-light form
    """ 
    with open(filename, "r") as dataset:
        rowList = [line.rstrip('\n') for line in dataset.readlines()]
        dataList = []
        newDataList = []
        
        #split each string into a list (commas between spaces)
        for row in rowList:
            row = row.split()
            dataList.append(row)
        
        #create tuple format [ (1, [(1, 1), (2, 1)]), (-1, [(1,0)]) ]
        for row in dataList:
            label = intOrFloat(row[0])
            vector = row[1:len(row)]
            
            tupVector = []

            for pairIndex in range(len(vector)):
                pair = vector[pairIndex]
                
                #find the colon to identify end of first digit
                c = 0
                while pair[c] != ':':
                    c += 1 
                digit1 = intOrFloat(pair[0:c])
                digit2 = intOrFloat(pair[c+1:len(pair)])
                
                #create (index, value) tuple out of index:value pair
                newPair = digit1, digit2
                tup = tuple(newPair)
                                
                #add tuple to new vector
                tupVector.append(tup)
                
            newRow = label, tupVector
            newDataList.append(tuple(newRow))
     
    return newDataList

dummy = object()

def listToMatlab(dataList, filename, dimension = 784):
    """
    converts dataList to matlab format and writes dataList to a text file with name 'filename'  

    inputs: dataList, a list of data in python svm-light format
            filename, the desired name for the file
            dimension, the dimension of the vector space

            Note: dimension for MNIST dataset is 784 (set as default)

    outputs: none (file is written with specified 'filename' in path)
    """
    zero = float(0)
    labels = []
    vectors = []
    lengths = []

    matrix = []

    for pair in dataList:
        label = pair[0]
        vector = pair[1]
        labels.append(label)
        vectors.append(vector)
        lengths.append(vector[len(vector)-1][0])

    length = max(lengths)

    for vector in vectors:
        i = 1
        ithRow = []
        for j in range(len(vector)):
            index = vector[j][0]
            value = vector[j][1]

            if index == i:
                ithRow.append(value)

            else:
                while i < index:
                    ithRow.append(zero)
                    i += 1
                ithRow.append(value)

            i += 1

        while i <= dimension: #length:
            ithRow.append(zero)
            i += 1
        
        matrix.append(ithRow)

    # print matrix, labels
    
    # write vectors
    with open(filename + '.txt', 'w+') as writeFile:
        for row in matrix:
            newRow = ''
            for i in range(len(row)):
                number = row[i]
                if i == len(row) - 1:
                    newRow += str(number)
                else:
                    newRow += str(number)+', '

            writeFile.write(newRow + '\n')
    # write labels
    with open(filename + '_labels.txt', 'w+') as labelFile:
        newRow = ''
        for i in range(len(labels)):
            number = labels[i]
            if i == len(labels) - 1:
                newRow += str(number)
            else:
                newRow += str(number)+', '

        labelFile.write(newRow)

    return 0


def labelParse(dataList, size, removeOrKeep, label, label2=dummy, label3=dummy, verbosity = 0):
    """ 
    creates a new list of length 'size', selecting only those vectors with 
    labels 'label', 'label2', and 'label3' in equal amounts
    Note: number must be divisible by 2 for 2 labels or 3 for 3 labels if an even
        partition is desired
    
    inputs: dataList, a list of data in python svm-light form
            size, the size of the desired subset
            removeOrKeep, a string: if 'r', dataList will be destructively altered 
                and will no longer contain the subset that was created
            label, the first desired label
            label2, the second desired label (optional--default value is 'label')
            label3, the third desired label (optional--default value is 'label')
            verbosity, 1 or 0 to print more or less (optional--default is 0)
            
    outputs: labelList, the specified subset of dataList randomly selected
            dataList, the original dataList either modified ('r') or preserved ('')
    """

    labelList = []
    
    #case variable to determine how to partition the subset
    case = 3
    
    #if user does not input label2 or label3, set redundant values
    if label3 is dummy:
        label3 = label
        case = 2
    if label2 is dummy:
        label2 = label3
        case = 1

    #set random sampling order
    randomOrder = list(range(len(dataList)))
    random.shuffle(randomOrder)
    
    if case == 1: #subset contains only one type of label
        size1 = size

        for i in randomOrder:
            entry = dataList[i]
            if size1 == 0:
                break
            if entry[0] == label:
                    labelList.append(entry)
                    size1 -= 1
    
                    if removeOrKeep == 'r':
                        dataList[i] = 0

                
    if case == 2: #subset contains two types of labels in equal amounts
        size1 = size / 2
        size2 = size1
        
        for i in randomOrder:
            entry = dataList[i]
            if size1 == 0 and size2 == 0:
                break
                
            if entry[0] == label:
                if size1 > 0:
                    labelList.append(entry)
                    size1 -= 1

                    if removeOrKeep == 'r':
                        dataList[i] = 0

            if entry[0] == label2:
                if size2 > 0:
                    labelList.append(entry)
                    size2 -= 1

                    if removeOrKeep == 'r':
                        dataList[i] = 0 
                
    if case == 3: #subset contains three types of labels in equal amounts
        size1 = size / 3
        size2 = size1
        size3 = size2
        
        for i in randomOrder:
            entry = dataList[i]
            if size1 == 0 and size2 == 0 and size3 == 0:
                break
                
            if entry[0] == label:
                if size1 > 0:
                    labelList.append(entry)
                    size1 -= 1

                    if removeOrKeep == 'r':
                        dataList[i] = 0

            if entry[0] == label2:
                if size2 > 0:
                    labelList.append(entry)
                    size2 -= 1

                    if removeOrKeep == 'r':
                        dataList[i] = 0

            if entry[0] == label3:
                if size3 > 0:
                    labelList.append(entry)
                    size3 -= 1

                    if removeOrKeep == 'r':
                        dataList[i] = 0
    
    if size1 != 0:
        print('FAILURE: Not enough vectors with label', label, 'for an even partition.')
        return 0

    if case == 2 and size2 != 0:
        print('FAILURE: Not enough vectors with label', label2, 'for an even partition.')
        return 0

    if case == 3 and size3 != 0:
        print('FAILURE: Not enough vectors with label', label3, 'for an even partition.')
        return 0
    
    if removeOrKeep == 'r':
        #remove labels marked for deletion
        dataList = [entry for entry in dataList if entry != 0]
    
    if verbosity != 0:
        if removeOrKeep == 'r':
                print("DESTRUCTIVELY created a subset containing:")
        else:
            print("Created a subset containing:")

        print(len(labelList)/case,"vectors with label", label)

        if case != 0:
            print(len(labelList)/case,"vectors with label", label2)

        if case == 3:
            print(len(labelList)/case,"vectors with label", label3)
        
    return labelList, dataList
        
    
def randomLabelRemove(percentToRemove, dataList):
    """
    removes percentToRemove of the labels from dataList randomly, 
    replacing them with zeros
    
    inputs: percentToRemove, the precentage of labels to be removed
            dataList, a list of data in python svm-light form
    outputs: a new list with the specified percentage of labels replaced by zero
    """
    #copy dataList so as not to alter original list
    newDataList = dataList
    
    numToRemove = ( float(percentToRemove) / 100) * len(newDataList)
    
    if numToRemove >= len(newDataList):
        return newDataList
        
    index = 0    
    while numToRemove != 0:
        choice = random.choice([0,1]) # 0 means make zero, 1 means skip
        
        if newDataList[index][0] != 0: #if entry is nonzero, make zero OR skip
            
            if choice == 0:
                label = 0
                vector = newDataList[index][1]
                
                entryList = label, vector
                entry = tuple(entryList)
                
                newDataList[index] = entry
                
                numToRemove -= 1
            
        index += 1
        index = index % len(newDataList)
        
    return newDataList
    
   
def intOrFloat(s):
    """ identifies whether a string is a float or an int and returns the correct type """
    try:
        return int(s)
    except ValueError:
        return float(s) 
        

def writeData(writeName, parseName, size, label, label2=dummy, label3=dummy):
    """ writes labelParse(dataset, size, label, label2, label3) to the file with name 'writeName'
        'label2' and 'label3' are optional variables
    """
    
    writeFile = open(writeName, 'w+')
    
    toAdd = labelParse(sparseToList(parseName), size, label, label2, label3)
    
    toAdd = replaceAll(toAdd, label2, -1)
    
    writeFile.write( str( toAdd ) )
    
    writeFile.close()
    
    print('DONE')


def replaceAll(dataset, oldLabel, newLabel):
    """ replaces all 'oldLabel' labels with 'newLabel' in dataset
    """
    newDataset = dataset
    if oldLabel == newLabel:
        return newDataset
    
    for i in range(len(newDataset)):
        
        if newDataset[i][0] == oldLabel:
            vector = newDataset[i][1]
            entryList = newLabel, vector
            newEntry = tuple(entryList)
            
            newDataset[i] = newEntry
            
    return newDataset
            
            
def labelCount(dataset, label):
    """ counts the number of vectors with label 'label' in dataset
    """  
    
    count = 0
    for entry in dataset:
        if entry[0] == label:
            count += 1
    print(count, 'vectors with label', label) 
    
    
def writeFile(dataset, filename, variableName):
    """ writes dataset to filename as variable 'variableName'
    """
    writeFile = open(filename, 'r+')
        
    writeFile.write( variableName + ' = ' + str( dataset ) )
    
    writeFile.close()
    
    print('DONE')

    
################################## SVM IMPLEMENTATION ########################################
    

def runSVMLight(trainName,testName, kerneltype, c_param = 1.0, gamma_param = 1.0, VERBOSITY = 0):
    """
    converts data to python format only if not already in python format 
    (files in python format are of type list, otherwise they are filenames)
    
    inputs: trainName, either the training data in svm-light format or the name of the training data file in LIBSVM/sparse format
            testName, either the test data in svm-light format or the name of the test data file in LIBSVM/sparse format
            kerneltype, (str)the type of kernel (linear, polynomial, sigmoid, rbf, custom)
            c_param, the C parameter (default 1)
            gamma_param, the gamma parameter (default 1)
            VERBOSITY, 0, 1, or 2 for less or more information (default 0)
    
    outputs: (positiveAccuracy, negativeAccuracy, accuracy)
    """
    if type(trainName) == list:
        trainingData = trainName
    else:
        trainingData = sparseToList(trainName)
        
    
    if type(testName) == list:
        testData = testName
    else:
        testData = sparseToList(testName)
        
    if VERBOSITY == 2:
        print("Training svm.......")

    # train a model based on the data
    ### Note: kerneltype must be a byte string ###
    model = svmlight.learn(trainingData, type='classification', verbosity=VERBOSITY, kernel=kerneltype.encode('utf-8'), C=c_param, rbf_gamma=gamma_param)


    ### Train without encoding kernel into byte string: ###
        # model = svmlight.learn(trainingData, type='classification', verbosity=VERBOSITY, kernel=kerneltype, C=c_param, rbf_gamma=gamma_param )
    

    # model data can be stored in the same format SVM-Light uses, for interoperability
    # with the binaries.
    
    # if type(trainName) == list:
    #     svmlight.write_model(model, time.strftime('%Y-%m-%d-')+datetime.datetime.now().strftime('%H%M%S%f')+'_model.dat')
    # else:
    #     svmlight.write_model(model, trainName[:-4]+'_model.dat')
    
    if VERBOSITY == 2:
        print("Classifying........")

    # classify the test data. this function returns a list of numbers, which represent
    # the classifications.
    predictions = svmlight.classify(model, testData)
    
    # for p in predictions:
    #     print '%.8f' % p
    
    correctLabels = correctLabelRemove(testData)

    # print 'Predictions:'
    # print predictions
    # print 'Correct Labels:'
    # print correctLabels

    return predictionCompare(predictions, correctLabels, VERBOSITY)
        
    #return predictions == rando('predictions_copy.txt')
    
    
################################## COMPARING RESULTS ########################################

def importCPredictions(filename):
    """ imports the predictions file that svm-light in C outputs and converts it to list form """
    with open(filename, "r") as dataset:
        dataList = [float(line.rstrip('\n')) for line in dataset.readlines()]
    return dataList


def correctLabelRemove(dataset):
    """ removes all the labels from a dataset and puts them in a list for 
        comparison to the SVM predictions list
        
        input: the tested dataset **without any labels removed**
        output: a list of labels for comparison to the 'predictions' list 
    """
    correctLabels = []
    for entry in dataset:
        correctLabels.append(entry[0])
    return correctLabels
    
def predictionCompare(predictions, correctLabels, verbosity = 0):
    """ compares the svm's predictions to the correctLabels
        
        input: predictions, the list used in runSVMLight
               correctLabels, the list of known correct labels 
        output: (positiveAccuracy, negativeAccuracy, accuracy), 3 percentage accuracies 
    """
    numberOfMistakes = 0
    incorrectPositives = 0
    incorrectNegatives = 0
    numberOfPoints = len(correctLabels)
    
    for i in range(len(predictions)):
        if float(predictions[i])*correctLabels[i] <= 0:
            numberOfMistakes += 1
            if correctLabels[i] < 0:
                incorrectPositives += 1
            else:
                incorrectNegatives += 1

    accuracy = 1 - float(numberOfMistakes)/numberOfPoints

    #count number of positive and negative labels
    positiveTotal = 0
    negativeTotal = 0
    for i in range(len(predictions)):
        if predictions[i] > 0:
            positiveTotal += 1
        else:
            negativeTotal += 1

    correctPositiveTotal = 0
    correctNegativeTotal = 0
    for i in range(len(correctLabels)):
        if correctLabels[i] > 0:
            correctPositiveTotal += 1
        else:
            correctNegativeTotal += 1

    #set values for zero case
    positiveAccuracy = 100.0
    negativeAccuracy = 100.0

    if correctPositiveTotal != 0:
        positiveAccuracy = (float(positiveTotal-incorrectPositives)/correctPositiveTotal)*100
    if correctNegativeTotal != 0:
        negativeAccuracy = (float(negativeTotal-incorrectNegatives)/correctNegativeTotal)*100
    
    accuracy *= 100
    
    if verbosity == 1:
        print('Accuracy on test set: ', accuracy, '%', '(', numberOfMistakes, 'misclassified,', numberOfPoints, 'total )')
    
    if verbosity == 2:
        if float(0) in predictions:
            print('Positive predictions: N/A due to zero valued predictions')
            print('Negative predictions: N/A due to zero valued predictions')
        else:
            print('Accuracy on positive labels: ', positiveAccuracy, '%', '(', incorrectPositives, 'incorrect positive labels,', positiveTotal - incorrectPositives, 'correct positive labels )')
            print('Accuracy on negative labels: ', negativeAccuracy, '%', '(', incorrectNegatives, 'incorrect negative labels,', negativeTotal - incorrectNegatives, 'correct negative labels )')
    
    return positiveAccuracy, negativeAccuracy, accuracy
