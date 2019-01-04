import EM
import utils

if __name__ == "__main__":

	topics = utils.load_topics(fname = "data/topics.txt")
	data = utils.read_file(topics, fname = "data/develop.txt")
	voc = utils.collect_vocab(data)
	ntk, num_of_words = utils.collect_n_tk(data, voc)
	W_t_i, likelihoods = EM.EM(data, voc, ntk, num_of_words, smoothing_const = 1, epsilon = 1e-3, minimal_change = 1e-3)
	utils.write_classifications(W_t_i, data)

# minimal change == when to end em loop (end condition)
# smoothing_const == lambda for lidston  change to 1
# epsilon == from the document when alpha_i is too small