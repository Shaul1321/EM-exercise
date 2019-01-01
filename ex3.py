import EM
import utils

if __name__ == "__main__":

	topics = utils.load_topics(fname = "data/topics.txt")
	data = utils.read_file(topics, fname = "data/develop.txt")
	voc = utils.collect_vocab(data)
	ntk = utils.collect_n_tk(data, voc)
	W_t_i, likelihoods = EM.EM(data, voc, ntk, smoothing_const = 1e-2, epsilon = 1e-3, minimal_change = 1e-3)
	utils.classify(W_t_i, data)
