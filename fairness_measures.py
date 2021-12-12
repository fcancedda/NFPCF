import torch
import numpy as np


# %% FAIRNESS CALCULATOR
def fairness_measures(model, df_val, num_items, protected_attributes, device):
    model.eval()
    user_input = torch.LongTensor(df_val['user_id'].values).to(device)
    item_input = torch.LongTensor(df_val['like_id'].values).to(device)
    y_hat = model(user_input, item_input)

    avg_epsilon = compute_edf(
        protected_attributes=protected_attributes,
        predictions=y_hat,
        n_classes=num_items,
        item_input=item_input,
        device=device
    )
    U_abs = compute_absolute_unfairness(protected_attributes, y_hat, num_items, item_input, device=device)

    avg_epsilon = avg_epsilon.cpu().detach().numpy().reshape((-1,)).item()
    print(f"average differential fairness: {avg_epsilon: .3f}")

    U_abs = U_abs.cpu().detach().numpy().reshape((-1,)).item()
    print(f"absolute unfairness: {U_abs: .3f}")


def compute_edf(protected_attributes, predictions, n_classes, item_input, device):
    # compute counts and probabilities
    S = np.unique(protected_attributes)  # number of genders: male = 0; female = 1
    counts_class_one = torch.zeros((n_classes, len(S)), dtype=torch.float).to(
        device)  # each entry corresponds to an intersection, arrays sized by largest number of values
    counts_total = torch.zeros((n_classes, len(S)), dtype=torch.float).to(device)

    concentration_parameter = 1.0
    dirichlet_alpha = concentration_parameter / n_classes

    for i in range(len(predictions)):
        counts_total[item_input[i], protected_attributes[i]] = counts_total[
                                                                   item_input[i],
                                                                   protected_attributes[i]] + 1.0
        counts_class_one[item_input[i], protected_attributes[i]] = counts_class_one[
                                                                       item_input[i], protected_attributes[i]
                                                                   ] + predictions[i]

    # probabilitiesClassOne = counts_class_one/counts_total
    probabilities_for_df_smoothed = (counts_class_one + dirichlet_alpha) / (counts_total + concentration_parameter)
    avg_epsilon = differential_fairness_multi_class(probabilities_for_df_smoothed, device)
    return avg_epsilon


def differential_fairness_multi_class(probabilities_of_positive, device):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilon_per_class = torch.zeros(len(probabilities_of_positive), dtype=torch.float).to(device)
    for c in range(len(probabilities_of_positive)):
        epsilon = torch.tensor(0.0).to(device)  # initialization of DF
        for i in range(len(probabilities_of_positive[c])):
            for j in range(len(probabilities_of_positive[c])):
                if i == j:
                    continue
                else:
                    # ratio of probabilities of positive outcome
                    epsilon = torch.max(
                        epsilon,
                        torch.abs(torch.log(probabilities_of_positive[c, i]) -
                                  torch.log(probabilities_of_positive[c, j])))
        epsilon_per_class[c] = epsilon  # overall DF of the algorithm
    avg_epsilon = torch.mean(epsilon_per_class)
    return avg_epsilon


def compute_absolute_unfairness(protected_attributes, predictions, n_classes, item_input, device):
    # compute counts and probabilities
    S = np.unique(protected_attributes)  # number of gender: male = 0; female = 1
    group_item_score = torch.zeros((n_classes, len(S)), dtype=torch.float).to(
        device)  # each entry corresponds to an intersection, arrays sized by largest number of values
    score_per_group = torch.zeros(len(S), dtype=torch.float).to(device)
    count_per_item = torch.zeros((n_classes, len(S)), dtype=torch.float).to(device)

    concentration_parameter = 1.0
    dirichlet_alpha = concentration_parameter / n_classes

    for i in range(len(predictions)):
        group_item_score[item_input[i], protected_attributes[i]] = group_item_score[
                                                                       item_input[i],
                                                                       protected_attributes[i]
                                                                   ] + predictions[i]
        count_per_item[item_input[i], protected_attributes[i]] = count_per_item[
                                                                     item_input[i],
                                                                     protected_attributes[i]
                                                                 ] + 1.0
        score_per_group[protected_attributes[i]] = score_per_group[protected_attributes[i]] + predictions[i]
    # probabilitiesClassOne = countsClassOne/countsTotal
    avg_score_per_group_per_item = (group_item_score + dirichlet_alpha) / (count_per_item + concentration_parameter)
    avg_score = score_per_group / torch.sum(count_per_item, axis=0)
    # torch.mean(avg_score_per_group_per_item,axis=0)
    difference = torch.abs(avg_score_per_group_per_item - avg_score)
    u_abs = torch.mean(torch.abs(difference[:, 0] - difference[:, 1]))
    return u_abs
