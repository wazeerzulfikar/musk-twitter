import pandas as pd 
import sys  
import math
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('txt_filename', type=str)

args = argparser.parse_args()


data = pd.read_csv(args.filename)
tweet_text = data["text"]
fav = data["favorite_count"]

with open(args.txt_filename, 'w') as f:
	for i in range(len(tweet_text)):
#		for i in range(int(math.log10(fav[i]))):
		f.write(tweet_text[i])

