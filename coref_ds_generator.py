import collections
from termcolor import colored
import os
import numpy as np 
import pickle

# dataset: [doc, corefs], data type: list, dictionary
# doc: [sent], sent: [word]
# corefs: {coref}, key: coref. id, value: list of (idx_sent, idx_word_from, idx_word_to)
def visual_doc(document):
	# inp: document[sents, sent_level_relationship, corefs]
	# out: [coref_by_sent, document with colored text]
	formats = []
	for style in range(8):
		for fg in range(30, 38):
			s1 = ''
			for bg in range(40, 48):
				if bg-fg==10:
					continue
				format = ';'.join([str(style), str(fg), str(bg)])
				formats.append(format)

	def color(s, id):
		return '\x1b[%sm %s \x1b[0m' % (formats[id], s)

	sents = document[0]
	corefs = document[2]

	coref_by_sent = {}	
	for coref_id in corefs:
		for (idx_sent, idx_word_from, idx_word_to) in corefs[coref_id]:
			 if idx_sent not in coref_by_sent:
			 	coref_by_sent[idx_sent] = [[coref_id, idx_word_from, idx_word_to]]
			 else:
			 	coref_by_sent[idx_sent].append([coref_id, idx_word_from, idx_word_to])

	# print(coref_by_sent)	

	# visualize document
	doc = ""
	max_id_coref = 0
	if len(corefs.keys())>0:
		max_id_coref = max(corefs.keys())

	for (sent_idx, sent) in enumerate(sents):		
		if sent_idx not in coref_by_sent:
			doc += "[" + ' '.join(sent) + "]"
		else:
			s = ""		
			for (word_idx, word) in enumerate(sent):
				is_mentioned_word = False

				for coref in coref_by_sent[sent_idx]:
					if coref[1] <= word_idx <= coref[2]:
						s += color(word, max_id_coref - coref[0])
						is_mentioned_word = True
						break
				if not is_mentioned_word:
					s += word + " "

			doc += "["+ s + "]"
	return [coref_by_sent, doc]

def sent_relationship(size, corefs):	
	# generate a binary matrix expressing the sentence-level relationships	
	# corefs: dictionary of conferences where key is conref_id and value is a list [sent_idx, word_idx_from, word_idx_to]
	m = np.zeros(shape=(size, size), dtype=np.int32)
	for coref_id in corefs:
		sents = corefs[coref_id]
		for i in range(len(sents)-1):
			for j in range(i+1, len(sents)):
				m[sents[i][0], sents[j][0]]= 1
				m[sents[j][0], sents[i][0]]= 1
	return m

def doc_process(filename):
	# inp: a file consisting of several documents
	# out: list of document [[sents, sent_relationship, corefs, coref_by_sent, colored text]]

	docs = []
	for line in open(file=filename, encoding="utf8", mode="r").readlines():
		if line.startswith("#begin document"):
			sents, text = [], []
			corefs = {}
			open_coref = []			
			continue

		if line.startswith("#end document"):
			docs.append([sents, sent_relationship(len(sents), corefs), corefs])
			continue

		items = line.strip().split()

		# a word in the sent
		if len(items)!=0:
			text.append(items[3])
			# if '|' in items[-1]:
			if items[-1]!="-":
				idx_sent = len(sents)
				idx_word = items[2]
				coref_ids = items[-1].split('|')
				for coref_id in coref_ids:
					if coref_id.startswith('('):
						if coref_id.endswith(')'): # (12)							
							idx = int(coref_id[1:-1])
							if idx not in corefs:
								corefs[idx] = [[idx_sent, int(idx_word), int(idx_word)]]
							else:
								corefs[idx].append([idx_sent, int(idx_word), int(idx_word)])
						else: # (12
							idx = int(coref_id[1:])
							open_coref.append([idx, idx_sent, idx_word])
					elif coref_id.endswith(')'): # 12)
						idx = int(coref_id[:-1])
						for i in range(len(open_coref)-1, -1, -1):
							if open_coref[i][0]==idx:
								if idx not in corefs:
									assert(idx_sent==open_coref[i][1])
									corefs[idx] = [[open_coref[i][1], int(open_coref[i][2]), int(idx_word)]]
								else:
									corefs[idx].append([open_coref[i][1], int(open_coref[i][2]), int(idx_word)])

								open_coref.pop(i)
								break
					else:
						print(["invalid format" ]*100)
			continue

		# the end of a sent
		if len(text)!=0:
			sents.append(text)
			text = []	

	for document in docs:
		document += visual_doc(document)
	return docs		

# docs = doc_process("../../coref/datasets/conll2012/train/data/english/annotations/bc/cctv/00/cctv_0002.v4_gold_conll")
# for (i, doc) in enumerate(docs):
# 	print("doc ", i)
# 	for c in doc:
# 		print(c)
# 		print("\n")


def generate_ds_for_sent_coref_model():
	docs = []
	for dirpath, dirs, files in os.walk("../../coref/datasets/conll2012/test/data/english/annotations"):
		for filename in files:
			fname = os.path.join(dirpath,filename)
			if fname.endswith("_gold_conll"):
				docs += doc_process(fname)

	x, y = [], []
	for doc in docs:
		x.append(doc[0])
		y.append(doc[1])

	pickle.dump([x, y], open('test.pkl', 'wb'))
	print(len(x), len(y))
	print("Finished")

# generate_ds_for_sent_coref_model()

def sent_level_dataset_statistic(filename):
	x, y = pickle.load(open(filename, "rb"))
	samples = len(x)
	assert samples==len(y)

	longest_doc = max([len(doc) for doc in x])
	longest_sent = max([max([len(sent) for sent in doc]) for doc in x])

	print(longest_doc)
	print(longest_sent)

# sent_level_dataset_statistic("test.pkl")

def dataset_statistic(folder):
	docs = []
	for dirpath, dirs, files in os.walk(folder):
		for filename in files:
			fname = os.path.join(dirpath,filename)
			if fname.endswith("_gold_conll"):
				docs += doc_process(fname)

	total_docs = len(docs)
	total_sents, total_words, total_corefs, longest_doc, longest_sent = 0, 0, 0, 0, 0	

	for doc in docs:
		longest_doc = max(longest_doc, len(doc[0]))
		longest_sent = max(longest_sent, max([len(sent) for sent in doc[0]]))
		total_sents += len(doc[0])
		total_words += sum([len(sent) for sent in doc[0]])
		total_corefs += len(doc[2])

	print("1) Total documents:", total_docs)	
	print("2) Total sentences:", total_sents)
	print("3) Total words:", total_words)

	print("4) The longest document has {} sentences".format(longest_doc))
	print("5) The longest sentence has {} words".format(longest_sent))

	avg_sent_doc = total_sents // total_docs
	avg_word_sent = total_words // total_sents
	print("6) Average number of sentences per document:", avg_sent_doc)	
	print("7) Average number of words per sentence:", avg_word_sent)
	avg_span_sent = (avg_word_sent + (avg_word_sent - 10)) * 5
	print("8) Average number of spans (with maximize 10 words) per sentence:", avg_span_sent)
	avg_pair_sents_doc = avg_sent_doc * (avg_sent_doc + 1) // 2
	print("9) Average number of pairs of sents. per document: ", avg_pair_sents_doc)
	avg_pair_spans_doc = (avg_span_sent**2) * avg_pair_sents_doc
	print("10) Average number of pairs of spans per document: ", avg_pair_spans_doc)
	print("11) Total number of pairs of spans in the dataset:", avg_pair_spans_doc * total_docs)
	print("="*20)

dataset_statistic("../../coref/datasets/conll2012/train/data/english/annotations")
dataset_statistic("../../coref/datasets/conll2012/development/data/english/annotations")
dataset_statistic("../../coref/datasets/conll2012/test/data/english/annotations")