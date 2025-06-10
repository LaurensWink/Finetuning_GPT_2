import random

def construct_number_tasks(file_name: str, templates: list[str], eval_funktion, examples_per_template = 1000):
    data_set = {
        "settings": {
            "name": file_name,
            "num_examples_per_template": examples_per_template,
            "input_templates": templates
        },
        "examples": {}
    }

    global_index = 0

    for question in templates:
        for _ in range(examples_per_template):
            global_index = global_index+1
            n1, n2 = random.sample(range(1000), 2)
            answer = eval_funktion(n1,n2)
            input = question.format(n1 = n1, n2 = n2)
            data_set["examples"][global_index]["input"] = input
            data_set["examples"][global_index]["metadata"]["n1"] = n1
            data_set["examples"][global_index]["metadata"]["n2"] = n2
            data_set["examples"][global_index]["metadata"]["answer"] = answer