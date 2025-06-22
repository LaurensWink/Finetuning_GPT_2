from pjs.corpus_construction import ammount_tasks, eval_more_letter, eval_less_letter, construct_number_tasks, eval_bigger_number, eval_smaller_number, first_alphabetically, eval_alphabetically_first, json_load, order_task, eval_first_elem, eval_last_elem, before_after_tasks, eval_word_after, eval_word_before
from pjs.german_corpus_data import tasks, diff_, diff_1, diff_3, first_alphabetically_data, first_alphabetically_different_first_letter_data, first_alphabetically_far_first_letter_data, first_alphabetically_same_first_letter_data, first_alphabetically_consecutive_first_letter_data

german_corpus_data = json_load("data/corpus_data_de/word_sentence_data.json")
word_list = german_corpus_data["word_list"]
sentence_list = german_corpus_data["sentence_list"]

construct_number_tasks("bigger_number", tasks["bigger_number"], eval_bigger_number)
construct_number_tasks("smaller_number", tasks["smaller_number"], eval_smaller_number)

first_alphabetically("first_alphabetically", tasks["first_alphabetically"], first_alphabetically_data, eval_alphabetically_first)
first_alphabetically("first_alphabetically_different_first_letter", tasks["first_alphabetically_different_first_letter"], first_alphabetically_different_first_letter_data, eval_alphabetically_first)
first_alphabetically("first_alphabetically_far_first_letter", tasks["first_alphabetically_far_first_letter"], first_alphabetically_far_first_letter_data, eval_alphabetically_first)
first_alphabetically("first_alphabetically_same_first_letter", tasks["first_alphabetically_same_first_letter"], first_alphabetically_same_first_letter_data, eval_alphabetically_first)
first_alphabetically("first_alphabetically_consecutive_first_letter", tasks["first_alphabetically_consecutive_first_letter"], first_alphabetically_consecutive_first_letter_data, eval_alphabetically_first)

order_task("first_letter", tasks["first_letter"], word_list, "word", eval_first_elem)
order_task("last_letter", tasks["last_letter"], word_list, "word", eval_last_elem)
order_task("first_word", tasks["first_word"], sentence_list, "sentence", eval_first_elem)
order_task("last_word", tasks["last_word"], sentence_list, "sentence", eval_last_elem)

before_after_tasks("word_after", tasks["word_after"], sentence_list, eval_word_after)
before_after_tasks("word_before", tasks["word_before"], sentence_list, eval_word_before)

ammount_tasks("less_letters_length_diff_1", tasks["less_letters_length_diff_1"], diff_1, eval_less_letter)
ammount_tasks("less_letters_length_diff_3plus", tasks["less_letters_length_diff_3plus"], diff_3, eval_less_letter)
ammount_tasks("less_letters", tasks["less_letters"], diff_, eval_less_letter)

ammount_tasks("more_letters_length_diff_1", tasks["more_letters_length_diff_1"], diff_1, eval_more_letter)
ammount_tasks("more_letters_length_diff_3plus", tasks["more_letters_length_diff_3plus"], diff_3, eval_more_letter)
ammount_tasks("more_letters", tasks["more_letters"], diff_, eval_more_letter)