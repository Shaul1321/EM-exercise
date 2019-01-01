import numpy as np

def calcualte_likelihood(z, k):

	""" 
	calcualte log likelihood of the data.
	z: len(data) X 9 matrix
	z[t,i] = ln(alpha[i]) + sum over each word k of (ntk[t,k] * ln(P_ik[i, k]))
	"""
	
	m =  np.max(z, axis = 1)
	diff =  np.e ** (z.T-m).T
	diff = np.where(diff < np.e ** (-k), 0, diff)

	return np.sum(m + np.log(np.sum(diff, axis = 1)))
	
	
def EM_init(data, voc, ntk, smoothing_const):

	"""
	initialzie EM parameters: alpha (p(xi) for each cluster i) and P_ik (p[wk|xi] for each word k and cluster i)
	alpha: a numpy array of length 9
	P_ik: a 2d numpy array, 9 X len(voc)
	"""
	
	alpha = np.zeros(9)
	alpha.fill(1./9)
	
	P_ik = np.zeros((9, len(voc)))
	P_ik.fill(smoothing_const)
	
	for j, article in enumerate(data):
	
		cluster = j % 9
		P_ik[cluster] += ntk[j]
		
	# normalize by dividing each row by its sum

	P_ik = P_ik/P_ik.sum(axis=1, keepdims=True)
	
	return alpha, P_ik

	
def EM(data, voc, ntk, smoothing_const = 1e-3, k = 10, epsilon = 1e-3, minimal_change = 1e-3):

	"""
	perform EM iterations until convergence.
	
	:params:
	
		data: a list of tuples (article_txt, gold topics)
		voc: a list of words (after filtering rare words)
		ntk: a 2d numpy array, n[t,k] = count(wk) in the th doc
		smoothing_const: lidstone smoothing constant
		k: approximation constant for numerical stability
		epsilon: smoothing constant for W_ti calculations.
		minimal_change: convergence condition
		
	:return:
	
		W_ti: a 2d numpy arrayp, W_ti[t,i] = P(cluster i|document t)
		likelihoods: a list of recorded likelihood values
	"""
	
	N = len(data)
	V = len(voc)
	W_ti = np.zeros((len(data), 9))
	alpha, P_ik = EM_init(data, voc, ntk, smoothing_const)
	
	prev_likelihood, likelihood = -np.inf, -np.inf
	counter = 0
	likelihoods = []

	while ((likelihood - prev_likelihood) > minimal_change) or counter == 0:

		counter += 1

		# ***E step***
	
		#create z
		# z[t,i] = ln(alpha[i]) + sum over each word k of (ntk[t,k] * ln(P_ik[i, k]))
		
		# dimensions:
			# z: len(data) X 9 matrix
			# ntk: len(data) X len(voc)
			# P_ik: 9 X len(voc)

		z = np.dot(ntk, np.log(P_ik).T) + np.log(alpha)
		new_likelihood = calcualte_likelihood(z, k)
		m =  np.max(z, axis = 1)

		unnormalized_w_ti = np.e**((z.T-m).T)
		unnormalized_w_ti = np.where(unnormalized_w_ti < np.e**(-k), 0, unnormalized_w_ti)
		W_ti = unnormalized_w_ti / unnormalized_w_ti.sum(axis = 1, keepdims = True)
		
		# ***M step***
		
		alpha = (np.sum(W_ti, axis = 0) + epsilon) / (N + epsilon * N)
		unnormalized_P_ik = np.dot(W_ti.T, ntk) + smoothing_const
		P_ik = unnormalized_P_ik / unnormalized_P_ik.sum(axis = 1, keepdims = True)
		
		likelihood, prev_likelihood = new_likelihood, likelihood
		print (likelihood)
		likelihoods.append(likelihood)
		
	return W_ti, likelihoods

