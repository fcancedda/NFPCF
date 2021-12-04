import importlib
import data
import math


def load():
    importlib.reload(data)
    importlib.import_module('data', package='AttributeData')
    importlib.import_module('data', package='TargetData')


def zero_model_parameters(model):
    # sets all parameters to zero

    params1 = model.named_parameters()
    params2 = model.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data - dict_params2[name1].data)

    model.load_state_dict(dict_params2)


def add_model_parameters(model1, model2):
    # Adds the parameters of model1 to model2

    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(param1.data + dict_params2[name1].data)

    model2.load_state_dict(dict_params2)


def sub_model_parameters(model1, model2):
    # Subtracts the parameters of model2 with model1

    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_(dict_params2[name1].data - param1.data)

    model2.load_state_dict(dict_params2)


def divide_model_parameters(model, f):
    # Divides model parameters except for the user embeddings with f
    params1 = model.named_parameters()
    params2 = model.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 != 'user_embedding.weight':
            dict_params2[name1].data.copy_(param1.data / f)
    model.load_state_dict(dict_params2)


# EVALUATOR 2 (RETURNS ALL K VALS <= K)
def get_hit_ratio(rank_list, true_item):
    for item in rank_list:
        if item == true_item:
            return 1
    return 0


def get_ndcg(rank_list, true_item):
    for i in range(len(rank_list)):
        item = rank_list[i]
        if item == true_item:
            return math.log(2) / math.log(i + 2)
    return 0