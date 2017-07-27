import csv

with open('musk_tweet.csv') as f:
	reader = csv.reader(f)
	musk_tweets = [row[4] for row in reader]

print(musk_tweets)