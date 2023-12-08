# -*- coding: utf-8 -*-
from predictor_models.utils import *
from settings.config_file import *


def write_results(best_model, test_performances_record):
    k = int(config["param"]["k"])
    predictor_metric = map_predictor_metrics()
    with open(
            f'results/result_details/{config["dataset"]["type_task"]}/{config["dataset"]["dataset_name"]}_n{int(config["param"]["budget"])}_K{k}.txt',
            'a+') as results:
        results.write(f'\n #############    Result report    (Time: {RunCode})  seed:{num_seed} #############\n')
        results.write(f'Dataset: {(config["dataset"]["dataset_name"])} \n')
        results.write(f'Type task: {config["dataset"]["type_task"]} \n')
        results.write(
            f'{config["param"]["search_metric"]} of the best sampled model: {config["results"][f"""{config["param"]["search_metric"]}_of_best_sampled_model"""]} \n')
        results.write(f'Search space reduction strategy: {config["param"]["search_space_reduction_strategy"]} \n')
        results.write(
            f' Initial Search space: #function={int(config["search_space"]["total_function"])} | # Option: {int(config["search_space"]["total_choices"])}| #Architectures:{int(config["search_space"]["size_sp"])}  \n')
        results.write(
            f' Final Search space: #function={int(config["search_space"]["Final_total_function"])} | # Option: {int(config["search_space"]["final_total_choices"])}| #Architectures:{int(config["search_space"]["final_size"])} \n')
        results.write(f'Type of predictor loss: {(config["param"]["encoding_method"])}\n')
        results.write(f'Node feature size: {(config["predictor"]["Predictor_model"])}\n')
        results.write(f'Predictor dataset type: {(config["predictor"]["predictor_dataset_type"])}\n')
        results.write(f'search metric: {config["param"]["search_metric"]}\n')
        results.write(f'Predictor metric: {(config["predictor"]["predictor_metric"])}\n')
        results.write(f'Predictor type of criterion: {(config["predictor"]["criterion"])}\n')
        results.write(f'Type of sampling : {config["param"]["type_sampling"]}\n')
        results.write(f'Total budget: {int(config["param"]["budget"])}\n')
        results.write(f'Number of samples: {int(config["param"]["N"])}\n')
        results.write(f'Number of TopK models: {int(config["param"]["k"])}\n')
        results.write(f'Best model: {best_model}\n')
        for metric in predictor_metric:
            results.write(f'predictor train {metric}= {float(config["results"][f"{metric}_train"])} \n')
            results.write(f'predictor val {metric}= {float(config["results"][f"{metric}_val"])} \n')
        for metric, performance in test_performances_record.items():
            results.write(f'best_{metric}= {float(config["results"][metric])}   \n')
        results.write(f'{"--" * 20} END {"--" * 20}\n\n ')

    print(f'\n #############    Result report    (Time: {RunCode})  seed:{num_seed} #############\n')
    print(f'Dataset: {(config["dataset"]["dataset_name"])} \n')
    print(f'Type task: {config["dataset"]["type_task"]} \n')
    print(
        f'{config["param"]["search_metric"]} of the best sampled model: {config["results"][f"""{config["param"]["search_metric"]}_of_best_sampled_model"""]} \n')
    print(f'Search space reduction strategy: {config["param"]["search_space_reduction_strategy"]} \n')
    print(
        f' Initial Search space: #function={int(config["search_space"]["total_function"])} | # Option: {int(config["search_space"]["total_choices"])}| #Architectures:{int(config["search_space"]["size_sp"])}  \n')
    print(
        f' Final Search space: #function={int(config["search_space"]["Final_total_function"])} | # Option: {int(config["search_space"]["final_total_choices"])}| #Architectures:{int(config["search_space"]["final_size"])} \n')
    print(f'size of the initial search space = : {int(config["search_space"]["size_sp"])}\n')
    print(f'Type of predictor loss: {(config["param"]["encoding_method"])}\n')
    print(f'Node feature size: {(config["predictor"]["Predictor_model"])}\n')
    print(f'Predictor dataset type: {(config["predictor"]["predictor_dataset_type"])}\n')
    print(f'search metric: {(config["param"]["search_metric"])}\n')
    print(f'Predictor metric: {(config["predictor"]["predictor_metric"])}\n')
    print(f'Predictor type of criterion: {(config["predictor"]["criterion"])}\n')
    print(f'Type of sampling : {config["param"]["type_sampling"]}\n')
    print(f'Total budget: {int(config["param"]["budget"])}\n')
    print(f'Number of samples: {int(config["param"]["N"])}\n')
    print(f'Number of TopK models: {int(config["param"]["k"])}\n')
    print(f'Best model: {best_model}\n')

    for metric in predictor_metric:
        print(f'predictor train {metric}= {float(config["results"][f"{metric}_train"])} \n')
        print(f'predictor val {metric}= {float(config["results"][f"{metric}_val"])} \n')

    for metric, performance in test_performances_record.items():
        print(f'best_{metric}= {float(config["results"][metric])} \n')

    print(f'{"--" * 20} END {"--" * 20}\n\n ')


def get_minutes(seconds):
    return round(seconds / 60, 2)


def get_hours(minutes):
    return round(minutes / 3600, 2)

