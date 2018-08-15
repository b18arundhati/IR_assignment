import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import sys
import sklearn
from sklearn import metrics
import operator

glove_dict = {}

import numpy as np

glove_file = 'glove.840B.300d.txt'
query_file = 'query.txt'
model_output = 'output.txt'

query_vector = {}
query_terms = {}

with open(query_file, 'r') as f:
	for line in f.read().split('\n'):
		if line == '':
			continue
		print "line: ",line
		print len(line.strip().split())
		qid = line.strip().split()[0]
		words = line.strip().split()[1:]
		query_terms[qid] = words
		query_vector[qid] = np.zeros((300,))
#sys.exit(0)
dic = []
with open(glove_file, 'r') as f:
	while True:
		line = f.readline()
		if line == '':
			break
		word = line.strip().split()[0]
		dic.append(word)
		for qid in query_terms:
			if word in query_terms[qid]:
				query_vector[qid] += np.array(line.strip().split()[1:], dtype='float64')

#expanding the query
expanded_query = {}
query_vector.keys().sort()
f1 = open('expanded_query.txt','a')
for qid in query_vector:
	print 'working on query: ',qid
	qvec = query_vector[qid] 
	top5 = []
	with open(glove_file, 'r') as f:
		while True:
			line = f.readline()
			if line == '':
				break
			word = line.strip().split()[0]
			vec = np.array(line.strip().split()[1:], dtype='float64')

			sim = metrics.pairwise.cosine_similarity(qvec.reshape(1,-1), vec.reshape(1,-1))
			top5.append((word, sim))
			top5 = sorted(top5, key=operator.itemgetter(1), reverse=True)
			if len(top5) > 5:
				top5.pop(-1)

	expanded_query[qid] = [w[0] for w in top5]	
	text = qid + '\t|\t' + ' '.join(query_terms[qid]) + '\t|\t' + ' '.join(expanded_query[qid]) + '\n'
	f1.write(text)

f1.close()

#print query_vector
print 'done'
#print len(dic)
#print dic	

#with open(glove_file, 'r') as f:
#	for line in f.read().split('\n'):
#		glove_dict[line.strip().split()[0]] = np.array(line.strip().split()[1:])

#print len(glove_dict)
