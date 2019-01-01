from collections import Counter, defaultdict
import numpy as np

def read_file(topics, fname = "data/develop.txt"):

	"""
	return: data, a list of tuples (text, categories).
		data[i] contain the info of the ith article:
		its text (as a string) and its topics (a list of strings).
	"""
	
	with open(fname, "r") as f:
	
		lines = f.readlines()
		
	lines = filter(lambda line: len(line) > 0, lines)
	data = []
	
	for i, line in enumerate(lines):
		
		if i % 2 == 0:
				if i > 0:
					data.append((text, article_topics))
					
				_, _, article_topics = line.strip().split("\t", 2)
				article_topics = [t for t in topics if t in article_topics]


		else:
			text = line.strip()
	
	return data
	
def load_topics(fname = "data/topics.txt"):
	
	topics = []
	
	with open(fname, "r") as f:
	
		lines = f.readlines()
		
	for line in lines:
		
		if line:
			
			topics.append(line.strip())
	
	return topics

def collect_vocab(data):
	"""
	return: the vocabulary (as a list of words), after filtering rare words
	
	"""	
	all_texts = " ".join([text for (text, categories) in data])
	word_counts = Counter(all_texts.split(" "))
	common_words = filter(lambda k: word_counts[k] > 3, word_counts)
	return common_words

def collect_n_tk(data, voc):

	"""
	collect n_t_k: the counts for the kth word in the tth doc. 
	return a numpy array n_tk.
	"""
	
	word_to_index = {w:i for (i,w) in enumerate(voc)}
	
	n_tk = np.zeros((len(data), len(voc)))
	
	for t, article in enumerate(data):
	
		text, _ = article
		
		words = text.split(" ")
		
		for w in words:
		
			k = word_to_index[w] if w in word_to_index else None
			
			if k:
				n_tk[t, k] += 1
			
	return n_tk

	
	
def write_classifications(W_t_i, data):

	num_articles, num_clusts = W_t_i.shape
	clust_to_topics = defaultdict(list)
	
	with open("clust.pred", "w") as f:
	
		for t in range(num_articles):
		
			clust = np.argmax(W_t_i[t])
			f.write(str(clust) + "\n")
			_, topics = data[t]
			clust_to_topics[clust].extend(topics)
	
	for (k,v) in clust_to_topics.items():

		topics_counter = Counter(v)
		topics =  topics_counter.items()
		topics_by_freq = sorted(topics, key = lambda (k,v): -v)
		topics_as_str = " ".join([topic + " : " + str(count) for (topic, count) in topics_by_freq])
		print
		print ("cluster: {}; topics: {}".format(str(k), topics_as_str))
		print
		print ("=======================================================")
	
	
	


