# -*- coding: utf-8 -*-
from predictor_models.utils import *
from settings.config_file import *
from accelerate import Accelerator
accelerator = Accelerator()




@accelerator.on_local_main_process
def write_results(best_model, test_performances_record):
    k = int(config["param"]["k"])
    search_metric = config["param"]["search_metric"]
    predictor_metric = map_predictor_metrics()
    with open(
            f'results/result_details/{config["dataset"]["type_task"]}/{config["dataset"]["dataset_name"]}_n{int(config["param"]["budget"])}_K{k}.txt',
            'a+') as results:
        results.write(f'\n #############    Result report    (Time: {RunCode})  #############\n')
        results.write(f'Dataset: {(config["dataset"]["dataset_name"])} \n')
        results.write(f'Type task: {config["dataset"]["type_task"]} \n')
        results.write(f'{config["param"]["search_metric"]} of the best sampled model: {config["results"][f"""{config["param"]["search_metric"]}_of_best_sampled_model"""]} \n')
        results.write(f'Search space reduction strategy: {config["param"]["search_space_reduction_strategy"]} \n')
        results.write(f' Initial Search space: #function={int(config["search_space"]["total_function"])} | # Option: {int(config["search_space"]["total_choices"])}| #Architectures:{int(config["search_space"]["size_sp"])}  \n')
        results.write(f' Final Search space: #function={int(config["search_space"]["Final_total_function"])} | # Option: {int(config["search_space"]["final_total_choices"])}| #Architectures:{int(config["search_space"]["final_size"])} \n')
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
        results.write(f'Best sampled model: {best_model}\n')
        results.write(f'Best Sampled model {search_metric}: {float(config["results"][f"{search_metric}_of_best_sampled_model"])}\n')
        results.write(f'f"Best_model_{search_metric}": {float(config["results"][f"Best_model_{search_metric}"])}\n')

        for metric in predictor_metric:
            results.write(f'predictor {metric}: |Train= {float(config["predictor"][f"{metric}_train"])} |Val={float(config["predictor"][f"{metric}_val"])} |Test={float(config["predictor"][f"{metric}_test"])} \n')
        for metric, performance in test_performances_record.items():
            results.write(f'stand_alone_best_{metric}= {float(config["results"][metric])}   \n')
        results.write(f'dataset_time_cost: {float(config["time"]["dataset_time_cost"])}\n')
        try:
            results.write(f'search_space_reduction_time_cost: {float(config["time"]["sp_reduce"])}\n')
        except:
            pass
        results.write(f'Search time cost: {float(config["time"]["search_time"])}\n')
        results.write(f'total search time cost: {float(config["time"]["total_search_time"])}\n')
        results.write(f'Number of GPUs: {float(config["param"]["nb_gpu"])}\n')
        results.write(f'{"--" * 20} END {"--" * 20}\n\n ')

    print(f'\n #############    Result report    (Time: {RunCode})  #############\n')
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
    print(f'Best sampled  model: {best_model}\n')
    print(f'Best Sampled model {search_metric}: {float(config["results"][f"{search_metric}_of_best_sampled_model"])}\n')
    print(f'f"Best_model_{search_metric}": {float(config["results"][f"Best_model_{search_metric}"])}\n')

    for metric in predictor_metric:
        print(f'predictor {metric}: |Train= {float(config["predictor"][f"{metric}_train"])} |Val={float(config["predictor"][f"{metric}_val"])} |Test={float(config["predictor"][f"{metric}_test"])} \n')
    for metric, performance in test_performances_record.items():
        print(f'stand_alone_best_{metric}= {float(config["results"][metric])} \n')
    print(f'dataset_time_cost: {float(config["time"]["dataset_time_cost"])}\n')
    print(f'Search time cost: {float(config["time"]["search_time"])}\n')
    try:
        print(f'search_space_reduction_time_cost: {float(config["time"]["sp_reduce"])}\n')
    except:
        pass
    print(f'total search time cost: {float(config["time"]["total_search_time"])}\n')
    print(f'Number of GPUs: {float(config["param"]["nb_gpu"])}\n')
    print(f'{"--" * 20} END {"--" * 20}\n\n ')


def get_minutes(seconds):
    return round(seconds / 60, 2)


def get_hours(minutes):
    return round(minutes / 3600, 2)

