'''
                                     Best K values function

The function initializes KNN models with different K values and find the best K value for

    R^2 coefficient score
    Precision score
    Recall score

'''
#
####################################################################################  Libraries
#
# Data manipulation tool
import pandas as pd
# Scientific computing, array
import numpy as np
# Data visualization
from matplotlib import pyplot as plt
# Theme to use with matplotlib
from jupyterthemes import jtplot
jtplot.style(theme='chesterish')
# K-Nearest Neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
# Model evaluation scores
from sklearn.metrics import accuracy_score, recall_score, precision_score
#
####################################################################################  The Function
#
def best_k_value(train_data, train_labels, test_data, test_labels, processed_test_labels, k_range, grid_name):
    '''
    The best_k_value() function:

    -Takes the arguments:
        train_data: list, numpy.array or pandas.Series data type
        train_labels: list, numpy.array or pandas.Series data type
        test_data: list, numpy.array or pandas.Series data type
        test_labels: list, numpy.array or pandas.Series data type
        processed_test_labels: list, numpy.array or pandas.Series data type
        k_range: integer data type, desired K range to evaluate
        grid_name: string data type, grid title and grid saving name
    - Initializes KNN models with k_range values
    - Predicts
    - Computes
        R^2 scores
        accuracy scores
        precision scores
        recall scores

    - Finds best K for
    - R^2 coefficient score
    - Precision score
    - Recall score
    - Outputs a one row three column graph grid
        R^2 coefficient, scores vs K values
        Precision scores vs K values
        Recall scores vs K value
    - Saves grid as a .png with the name of the grid_name string value

    - returns
        a best K DataFrame
        a evaluation K scores DataFrame
    '''
    #
    # --------------------------------------------- Computes evaluation scores
    # Initializes Lists
    scores = []
    accuracies = []
    precisions = []
    recalls = []
    # K values loop
    for k in range(1, k_range):
        # Initializes model with the K value
        classifier = KNeighborsClassifier(n_neighbors = k)
        # Train model
        classifier.fit(train_data, train_labels)
        # R^2 score
        scores.append(classifier.score(test_data, test_labels))
        # Predicts
        predictions = classifier.predict(test_data)
        # Preprocesses 2d predictions
        processed_predictions = [tweet[0] for tweet in predictions] + [tweet[1] for tweet in predictions]
        # Computes evaluation scores
        accuracies.append(accuracy_score(processed_test_labels, processed_predictions))
        precisions.append(precision_score(processed_test_labels, processed_predictions))
        recalls.append(recall_score(processed_test_labels, processed_predictions))
    #
    # --------------------------------------------- Graph Grid
    fig = plt.figure(figsize=(21, 6))
    fig.suptitle(f'{grid_name}', fontsize=18, color='lightgray')
    # Space between graphs
    plt.subplots_adjust(wspace=0.37)
    # Initializes Lists to be used with graphs
    eval_scores = [scores, precisions, recalls]
    eval_names = ['R^2 Score', 'Precision', 'Recall']
    # Initializes best K dataframe
    df_best_k = pd.DataFrame(columns=['Evaluation', 'best_k', 'R^2 Score', 'Precision Score', 'Recall Score', 'Accuracy Score'])
    # Grid
    for i in range(3):
        # Plot grid location
        plt.subplot(1, 3, i+1)
        # Index of the highest score
        index_best_k = eval_scores[i].index(max(eval_scores[i][1:]))
        # Computing the ticks' step value to be render the x axis
        if int(k_range/11)%2 == 0:
            step = int(k_range/11)
        else:
            step = int(k_range/11)+1
        # Plot
        plt.plot(range(1,k_range), eval_scores[i], color='cyan')
        # Best K vertical line
        plt.axvline(index_best_k+1, linestyle=':', color='magenta')
        # x axis ticks
        plt.xticks(np.arange(0, k_range, step=step))
        # Plot titile and labels
        plt.title(f'Best K Value For {eval_names[i]}')
        plt.ylabel(eval_names[i])
        plt.xlabel('K')
        # ----- Text score box
        text_label = f'Best K for {eval_names[i]} = {index_best_k+1}\nScore = {round(scores[index_best_k], 3)}\nAccuracy = {round(accuracies[index_best_k], 3)}\nPrecision = {round(precisions[index_best_k], 3)}\nRecall = {round(recalls[index_best_k], 3)}'
        # Computes the x and y axis text box location
        y = max(eval_scores[i]) - (((max(eval_scores[i]) - min(eval_scores[i]))/2)/2)*2
        x = index_best_k + 1 + step/2
        # The box
        plt.text(x, y,
                 text_label,
                 horizontalalignment='left',
                 verticalalignment='center',
                 bbox=dict(facecolor='#323a47', edgecolor='None', alpha=0.9))
        # Best k
        df_best_k = df_best_k.append({'Evaluation':eval_names[i],
                                'best_k':index_best_k+1,
                                'R^2 Score':round(scores[index_best_k], 3),
                                'Precision Score':round(precisions[index_best_k], 3),
                                'Recall Score':round(recalls[index_best_k], 3),
                                'Accuracy Score':round(accuracies[index_best_k], 3)}, ignore_index=True)
    #
    # Saves grid
    plt.savefig(f'graph/{grid_name}.png')
    plt.show()
    # Evaluation score dataframe
    df_eval = pd.DataFrame({'K':range(1, k_range),
                            'Scores':scores,
                            'Accuracies':accuracies,
                            'Precisions':precisions,
                            'Recalls':recalls})

    return df_best_k, df_eval