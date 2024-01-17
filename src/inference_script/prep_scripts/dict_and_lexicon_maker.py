# Usage: python dict_and_lexicon_maker.py --wrd <path to train.wrd>train.wrd --lexicon <destnation path>/lexicon.lst --dict <destnation path>/dict.ltr.txt # noqa: E501
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=W0311, C0304, C0114, C0411, W0611

import numpy as np
import argparse

def main():
	parser = argparse.ArgumentParser()  # noqa: W191

	parser.add_argument("--wrd", type = str, help = "path to wrd file")  # noqa: W191,E251
	parser.add_argument("--lexicon", default = 'lexicon.lst', type = str, help = "path to wrd file")  # noqa: W191,E251
	parser.add_argument("--dict", default = 'dict.ltr.txt', type = str, help = "path to wrd file")  # noqa: W191,E251

	args = parser.parse_args()  # noqa: W191

	text_lines=[]  # noqa: W191,E225

	with open(args.wrd, mode = 'r', encoding='utf-8') as file_local:  # noqa: W191,E251
		text_lines = file_local.readlines()  # noqa: W191

	total_words = " ".join(text_lines).split(" ")  # noqa: W191
	total_words = [local_word.strip() for local_word in total_words]  # noqa: W191
	unique_words = np.unique(total_words)  # noqa: W191

	unique_characters_dict = []  # noqa: W191

	with open(args.lexicon, mode='w+', encoding='utf-8') as file_lexicon:  # noqa: W191
		for local_word in unique_words:  # noqa: W191
			if local_word != "":  # noqa: W191
				unique_characters = list(local_word)  # noqa: W191
				unique_characters_dict.extend(unique_characters)  # noqa: W191
				print(local_word + "\t" +  " ".join( unique_characters ) + " |", file=file_lexicon)  # noqa: W191,E222,E201,E202
	# noqa: W191
	print("** Lexicon File Created")  # noqa: W191


	unique_character_set =['|']  # noqa: W191,E303,E225
	unique_character_set.extend( np.unique(unique_characters_dict) )  # noqa: W191,E201,E202
	print("** Dictionary of length ", len(unique_character_set)," created as:")  # noqa: W191,E231
	print(unique_character_set)  # noqa: W191,W291 


	total_character_set = []  # noqa: W191,E303
	for word in total_words:  # noqa: W191
		characters = list(word)  # noqa: W191
		total_character_set.extend(characters)  # noqa: W191

	with open(args.dict, mode='w+', encoding='utf-8') as file_dict:  # noqa: W191
		for i in unique_character_set:  # noqa: W191
			print(i +" "+ str(total_character_set.count(i)), file=file_dict)  # noqa: W191,E225


if __name__ == "__main__":
	main()  # noqa: W191,W292,W291  
