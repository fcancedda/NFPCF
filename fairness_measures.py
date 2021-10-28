import torch
import numpy as np


class Measures:
    def __init__(self):
        pass

    def computeEDF(self, protected_attributes, predictions, numClasses, item_input, device):
        # compute counts and probabilities
        S = np.unique(protected_attributes)  # number of genders: male = 0; female = 1
        countsClassOne = torch.zeros((numClasses, len(S)), dtype=torch.float).to(
            device)  # each entry corresponds to an intersection, arrays sized by largest number of values
        countsTotal = torch.zeros((numClasses, len(S)), dtype=torch.float).to(device)

        concentrationParameter = 1.0
        dirichletAlpha = concentrationParameter / numClasses

        for i in range(len(predictions)):
            countsTotal[item_input[i], protected_attributes[i]] = countsTotal[
                                                                      item_input[i],
                                                                      protected_attributes[i]] + 1.0
            countsClassOne[item_input[i], protected_attributes[i]] = countsClassOne[
                                                                         item_input[i], protected_attributes[i]
                                                                     ] + predictions[i]

        # probabilitiesClassOne = countsClassOne/countsTotal
        probabilities_for_df_smoothed = (countsClassOne + dirichletAlpha) / (countsTotal + concentrationParameter)
        avg_epsilon = self.differentialFairnessMultiClass(probabilities_for_df_smoothed, device)
        return avg_epsilon

    @staticmethod
    def differentialFairnessMultiClass(probabilities_of_positive, device):
        # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
        # output: epsilon = differential fairness measure
        epsilonPerClass = torch.zeros(len(probabilities_of_positive), dtype=torch.float).to(device)
        for c in range(len(probabilities_of_positive)):
            epsilon = torch.tensor(0.0).to(device)  # initialization of DF
            for i in range(len(probabilities_of_positive[c])):
                for j in range(len(probabilities_of_positive[c])):
                    if i == j:
                        continue
                    else:
                        epsilon = torch.max(epsilon, torch.abs(torch.log(probabilities_of_positive[c, i]) - torch.log(
                            probabilities_of_positive[c, j])))  # ratio of probabilities of positive outcome
            # epsilon = torch.max(epsilon,torch.abs((torch.log(1-probabilitiesOfPositive[c,i]))-(torch.log(
            # 1-probabilitiesOfPositive[c,j])))) # ratio of probabilities of negative outcome
            epsilonPerClass[c] = epsilon  # overall DF of the algorithm
        avg_epsilon = torch.mean(epsilonPerClass)
        return avg_epsilon

    @staticmethod
    def compute_absolute_unfairness(protected_attributes, predictions, numClasses, item_input, device):
        # compute counts and probabilities
        S = np.unique(protected_attributes)  # number of gender: male = 0; female = 1
        score_per_group_per_item = torch.zeros((numClasses, len(S)), dtype=torch.float).to(
            device)  # each entry corresponds to an intersection, arrays sized by largest number of values
        score_per_group = torch.zeros(len(S), dtype=torch.float).to(device)
        count_per_item = torch.zeros((numClasses, len(S)), dtype=torch.float).to(device)

        concentration_parameter = 1.0
        dirichlet_alpha = concentration_parameter / numClasses

        for i in range(len(predictions)):
            score_per_group_per_item[item_input[i], protected_attributes[i]] = score_per_group_per_item[
                                                                               item_input[i],
                                                                               protected_attributes[i]
                                                                               ] + predictions[i]
            count_per_item[item_input[i], protected_attributes[i]] = count_per_item[
                                                                       item_input[i],
                                                                       protected_attributes[i]
                                                                     ] + 1.0
            score_per_group[protected_attributes[i]] = score_per_group[protected_attributes[i]] + predictions[i]
        # probabilitiesClassOne = countsClassOne/countsTotal
        avg_score_per_group_per_item = (score_per_group_per_item + dirichlet_alpha) / (count_per_item + concentration_parameter)
        avg_score = score_per_group / torch.sum(count_per_item, axis=0)
        # torch.mean(avg_score_per_group_per_item,axis=0)
        difference = torch.abs(avg_score_per_group_per_item - avg_score)
        u_abs = torch.mean(torch.abs(difference[:, 0] - difference[:, 1]))
        return u_abs
