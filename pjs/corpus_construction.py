import random
from loguru import logger

def construct_number_tasks(file_name: str, templates: list[str], eval_funktion, examples_per_template = 1000):
    '''all number_tasks type tasks'''
    global_index = 0
    data_set = {
        "settings": {
            "name": file_name,
            "num_examples_per_template": examples_per_template,
            "input_templates": templates
        },
        "examples": {}
    }

    for question in templates:
        for _ in range(examples_per_template):
            global_index = global_index+1
            n1, n2 = random.sample(range(1000), 2)
            answer = eval_funktion(n1,n2)
            input = question.format(n1 = n1, n2 = n2)
            data_set["examples"][global_index] = {}
            data_set["examples"][global_index]["input"] = input
            data_set["examples"][global_index]["metadata"]["n1"] = n1
            data_set["examples"][global_index]["metadata"]["n2"] = n2
            data_set["examples"][global_index]["metadata"]["answer"] = answer

def first_alphabetically(file_name: str, templates: list[str], word_list: list[list[str], list[str]], eval_funktion, examples_per_template = 1000):
    '''all first_alphabetically type tasks'''
    global_index = 0
    word_tupel_list = []
    data_set = {
        "settings": {
            "name": file_name,
            "num_examples_per_template": examples_per_template,
            "input_templates": templates
        },
        "examples": {}
    }

    for index_first in range(len(word_list[0])):
        for index_second in range(len(word_list[1])):
            word_tupel_list.append((word_list[0][index_first], word_list[1][index_second])) 

    if not len(word_tupel_list) == examples_per_template*len(templates):
        logger.error(f'word_tupel_list is smaller than example amount!')
        return

    for question in templates:
        for _ in range(examples_per_template):
            n1, n2 = random.sample(range(2), 2)
            word1 = word_tupel_list[global_index][n1]
            word2 = word_tupel_list[global_index][n2]
            global_index = global_index+1
            answer = eval_funktion(word1,word2)
            input = question.format(word1 = word1, word2 = word2)
            data_set["examples"][global_index] = {}
            data_set["examples"][global_index]["input"] = input
            data_set["examples"][global_index]["metadata"]["word1"] = word1
            data_set["examples"][global_index]["metadata"]["word2"] = word2
            data_set["examples"][global_index]["metadata"]["answer"] = answer

def order_task(file_name: str, templates: list[str], task_data: list[list[str]], task_type: str, eval_funktion, examples_per_template = 1000):
    '''first_letter/first_word/last_letter/last_word tasks (task_type can be "word" or "sentence")'''
    if task_type not in ("word", "sentence"):
        raise ValueError("Invalide task_type. Only 'word' or 'sentence' are accepted.")
    
    global_index = 0
    data_set = {
        "settings": {
            "name": file_name,
            "num_examples_per_template": examples_per_template,
            "input_templates": templates
        },
        "examples": {
        }
    }

    for question in templates:
        for _ in range(examples_per_template):
            subject = task_data[global_index]
            global_index = global_index+1
            answer = eval_funktion(subject)
            input = question.format(**{task_type: subject})
            data_set["examples"][global_index] = {}
            data_set["examples"][global_index]["input"] = input
            data_set["examples"][global_index]["metadata"][task_type] = subject
            data_set["examples"][global_index]["metadata"]["answer"] = answer

def ammount_tasks(file_name: str, templates: list[str], task_data: list[(list[str]), (list[str])], eval_funktion, examples_per_template = 1000):
    '''all more_/less_letters tasks'''
    global_index = 0
    word_tupel_list = []
    data_set = {
        "settings": {
            "name": file_name,
            "num_examples_per_template": examples_per_template,
            "input_templates": templates
        },
        "examples": {
        }
    }

    for tupel in task_data:
        for word1 in tupel[0]:
            for word2 in tupel[1]:
                word_tupel_list.append((word1, word2))

    for question in templates:
        for _ in range(examples_per_template):
            n1, n2 = random.sample(range(2), 2)
            word1 = word_tupel_list[global_index][n1]
            word2 = word_tupel_list[global_index][n2]
            global_index = global_index+1
            answer = eval_funktion(word1,word2)
            input = question.format(word1 = word1, word2 = word2)
            data_set["examples"][global_index] = {}
            data_set["examples"][global_index]["input"] = input
            data_set["examples"][global_index]["metadata"]["word1"] = word1
            data_set["examples"][global_index]["metadata"]["word2"] = word2
            data_set["examples"][global_index]["metadata"]["answer"] = answer

def before_after_tasks(file_name: str, templates: list[str], task_data: list[list[str]], eval_funktion, examples_per_template = 1000):
    '''word_before/word_after tasks'''
    global_index = 0
    data_set = {
        "settings": {
            "name": file_name,
            "num_examples_per_template": examples_per_template,
            "input_templates": templates
        },
        "examples": {
        }
    }

    for question in templates:
        for _ in range(examples_per_template):
            sentence = task_data[global_index]
            global_index = global_index+1
            qurey = eval_funktion(sentence)
            input = question.format(sentence = sentence)
            data_set["examples"][global_index] = {}
            data_set["examples"][global_index]["input"] = input
            data_set["examples"][global_index]["metadata"]["sentence"] = sentence
            data_set["examples"][global_index]["metadata"]["qurey"] = qurey