import numpy as np 
import tensorflow as tf 
import pickle
from gensim.models import KeyedVectors
import string
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d


class sent_model:
	def __init__(self, params=None):
		self.params = params

		self.batch_size = params["batch_size"]
		self.keep_prob = params["keep_prob"]
		self.nb_epochs = params["nb_epochs"]

		# load word and char dictionaries
		dicts = pickle.load(open(params["dicts_file"], "rb"))
		self.w2i = dicts["w2i"]
		self.i2w = dicts["i2w"]
		self.c2i = dicts["c2i"]
		self.i2c = dicts["i2c"]		
		print("Lengths of word and char dictionaries: {}, {}".format(len(self.w2i), len(self.c2i)))

		self.word_dim = params["word_dim"]
		self.word_vocab_size = len(self.w2i)
		self.char_dim = params["char_dim"]
		self.char_vocab_size = len(self.c2i)
		self.lstm_num_units = params["lstm_num_units"]		

		# load word embedding
		self.word_emb = np.zeros(shape=(self.word_vocab_size, self.word_dim))
		if "word_emb" in params:
			self.load_word_emb(params["word_emb"])

		tf.reset_default_graph()
		
		# [batch size, nb_words]
		self.tf_word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="word_ids")
		# real word lengths in sents
		self.tf_sentence_lengths= tf.placeholder(dtype=tf.int32, shape=[None], name="sentence_lengths")

		# [batch size, nb_words, nb_chars]
		self.tf_char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="char_ids")

		# output matrix with batch size: [batch size, nb_sents, nb_sents]		
		self.tf_target_matrix = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="target_matrix")

		# keep_prob
		self.tf_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")

		# learning rate
		self.tf_learning_rate= tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

		
		# load word embedding
		with tf.variable_scope("word_embedding"):
			tf_word_embeddings = tf.Variable(self.word_emb, dtype=tf.float32,
				trainable=True, name="word_embedding")
			embedded_words = tf.nn.embedding_lookup(tf_word_embeddings, self.tf_word_ids, name="embedded_words")
			self.input = embedded_words


		# CNN network to capture character-level features
		with tf.variable_scope("char_cnn"):
			tf_char_embeddings = tf.get_variable(name="char_embeddings",
												 dtype=tf.float32,
												 shape=[self.char_vocab_size, self.char_dim],
												 trainable=True,
												 initializer=xavier_initializer())

			embedded_cnn_chars = tf.nn.embedding_lookup(tf_char_embeddings,
															self.tf_char_ids,
															name="embedded_cnn_chars")

			conv1 = tf.layers.conv2d(inputs=embedded_cnn_chars,
										filters=self.params["conv1"][1],
										kernel_size=(1, self.params["conv1"][0]),
										strides=(1, 1),
										padding="same",
										name="conv1",
										kernel_initializer=xavier_initializer_conv2d())
			conv2 = tf.layers.conv2d(inputs=conv1,
										filters=self.params["conv2"][1],
										kernel_size=(1, self.params["conv2"][0]),
										strides=(1, 1),
										padding="same",
										name="conv2",
										kernel_initializer=xavier_initializer_conv2d())
			conv3 = tf.layers.conv2d(inputs=conv2,
										filters=self.params["conv3"][1],
										kernel_size=(1, self.params["conv3"][0]),
										strides=(1, 1),
										padding="same",
										name="conv3",
										kernel_initializer=xavier_initializer_conv2d())

			char_cnn = tf.nn.dropout(tf.reduce_max(conv3, axis=2), self.tf_keep_prob)

			# [sent, word, features]
			self.input = tf.concat([self.input, char_cnn], axis=-1)

		# Bi-LSTM to generate final input representation in combination with both left and right contexts
		with tf.variable_scope("bi_lstm_words"):
			cell_fw = tf.contrib.rnn.LSTMCell(self.params["lstm_num_units"])
			cell_bw = tf.contrib.rnn.LSTMCell(self.params["lstm_num_units"])
			(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.input,
																		sequence_length=self.tf_sentence_lengths,
																		dtype=tf.float32)
			# [sents, words, 2*lstm_units]
			self.bilstm_output = tf.concat([output_fw, output_bw], axis=-1)

		self.a = tf.reduce_max(self.bilstm_output, axis=1)

		# pad document in order to all document in the batch have the same length (nb_sents)
		self.tf_doc_boundaries = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

		# split a into list of original documents: batch_size documents with shape: [sents_in_document, 2*lstm_units]
		self.b = tf.split(self.a, self.tf_doc_boundaries, 0)
		self.nb_sents_in_longest_doc = tf.reduce_max(self.tf_doc_boundaries)

		# pad document with sents of <PAD> words in order to all documents have the same length: nb_sents
		self.c = tf.stack([tf.pad(self.b[i], [[0, self.nb_sents_in_longest_doc-self.tf_doc_boundaries[i]], [0, 0]], constant_values=self.w2i["<PAD>"]) for i in range(len(self.b))])		

		# [nb_documents, nb_sents, 2*lstm_units]
		self.c = tf.reshape(self.c, (self.batch_size, -1, 2*self.lstm_num_units))		

		
		# self.d = tf.expand_dims(self.c, 1)
		# self.e = tf.tile(self.d, (1, tf.shape(self.c)[1], 1, 1))

		# self.f = tf.expand_dims(self.c, 2)
		# self.g = tf.tile(self.f, (1, 1, tf.shape(self.c)[1], 1))

		# # [nb_documents, nb_sents, nb_sents, 4*lstm_units]
		# self.h = tf.concat([self.e, self.g], axis=-1)

		with tf.variable_scope("self_att"):
			x = self.c
			# q = tf.placeholder(dtype=tf.float32, shape=[dq, 1])
			de = x.get_shape()[-1]
			w = tf.get_variable(dtype=tf.float32, shape=[de], trainable=True, name="w")
			w1 = tf.get_variable(dtype=tf.float32, shape=[de, de], trainable=True, name="W1")
			w2 = tf.get_variable(dtype=tf.float32, shape=[de, de], trainable=True, name="W2")
			b1 = tf.get_variable(dtype=tf.float32, shape=[de], trainable=True, name="b1")
			b = tf.get_variable(dtype=tf.float32, shape=[], trainable=True, name="b")

			e1 = tf.transpose(tf.tensordot(x, w1, axes=1), [1,0,2]) #b, n, de -> n, b, de = n, [b,de]
			# print('e1', e1)
			e2 = tf.transpose(tf.tensordot(x, w2, axes=1), [1,0,2]) #b, n, de -> n, b, de
			# print('e2', e2)
			# tong = tf.map_fn(lambda i: i+ e2, e1)#
			tong = tf.transpose(tf.map_fn(lambda i: i + e2 + b1, e1), [2,0,1,3]) # b, n, n, de
			# print('tong', tong)

			weight = tf.nn.softmax(tf.tensordot(tf.tanh(tong), w, axes=1) + b) # b, n, n
			# print('weight:', weight)
			self.h = tf.transpose(tf.map_fn(lambda y: y*x,tf.expand_dims(tf.transpose(weight, [1,0,2]), -1)), [1,0,2,3])


		# with tf.variable_scope("attention"):
		# 	d = tf.tile(tf.expand_dims(self.c, 1), (1, tf.shape(self.c)[1], 1, 1))
		# 	print("d (expand_dims and tile): ",d)
		# 	f = tf.tile(tf.expand_dims(self.c, 2), (1, 1, tf.shape(self.c)[1], 1))
		# 	print("f (expand_dims and tile:",f)
		# 	i = tf.concat([d, f], 3)
		# 	print("i (concatenate):",i)
		# 	j = tf.layers.dense(inputs=i,
		# 						units=256, # dimenson of attention layer
		# 						activation=None, # Linear activation
		# 						kernel_initializer=xavier_initializer()) #, self.tf_dropout)
		# 	print("j (dense 256):", j)
			
		# 	# CNN
		# 	# conv3 = tf.layers.conv2d(inputs=j,
		# 	# 						 filters=100,
		# 	# 						 kernel_size=(3, 3),
		# 	# 						 strides=(1, 1),
		# 	# 						 padding="same",
		# 	# 						 kernel_initializer=xavier_initializer_conv2d(),
		# 	# 						 name="conv3")
		# 	# conv4 = tf.layers.conv2d(inputs=conv3,
		# 	# 						 filters=100,
		# 	# 						 kernel_size=(3, 3),
		# 	# 						 strides=(1, 1),
		# 	# 						 padding="same",
		# 	# 						 kernel_initializer=xavier_initializer_conv2d(),
		# 	# 						 name="conv4")

		# 	k = tf.layers.dense(inputs=j, #tf.concat([conv3, conv4], axis=-1),
		# 	# k = tf.nn.dropout(tf.layers.dense(inputs=conv4,
		# 									  units=1,
		# 									  activation=None, #tf.tanh,
		# 									  kernel_initializer=xavier_initializer())
		# 	print("k (dense 1):", k)
		# 	l = tf.squeeze(k, axis=[-1])
		# 	print("l (squeeze) :", tf.shape(l))
		# 	m = tf.nn.softmax(l, dim=-1)
		# 	print("m (softmax):", tf.shape(m))
		# 	n = tf.expand_dims(m, -1)
		# 	print("n (expand_dims): ", tf.shape(n))
		# 	self.h = n * f
						

		# self.k = tf.nn.dropout(tf.layers.dense(inputs=self.h, units=2*self.lstm_num_units, activation=None), self.tf_keep_prob)

		self.logits = tf.layers.dense(inputs=self.h, units=2, activation=None)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.tf_target_matrix, 2),
													   logits=self.logits,
													   name="loss_function"))

		# self.loss = tf.reduce_sum(self.cross_entropy) / tf.cast(tf.size(self.logits), tf.float32)

		self.pred = tf.argmax(self.logits, axis=-1)
		eq = tf.cast(tf.equal(tf.cast(self.pred, tf.int32), self.tf_target_matrix), tf.float32)
		self.acc = tf.reduce_mean(eq)

		self.opt = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate).minimize(self.loss)
		
		
	def train(self, train_pkl_file, dev_pkl_file, test_pkl_file):
		[raw_x_train, raw_y_train] = pickle.load(open(train_pkl_file, "rb"))
		[raw_x_dev, raw_y_dev] = pickle.load(open(dev_pkl_file, "rb"))
		[raw_x_test, raw_y_test] = pickle.load(open(test_pkl_file, "rb"))

		# index words: sample -> sent -> idx_word
		idx_words_train = [[[self.w2i[word.lower()] if word.lower() in self.w2i else self.w2i["<UNK>"] for word in sent]
							for sent in sample] 
							for sample in raw_x_train]

		# index characters: sample -> sent -> word -> idx_char
		idx_chars_train = [[[[self.c2i[char] if char in self.c2i else self.c2i["<UNK>"] for char in word] for word in sent]
							for sent in sample] 
							for sample in raw_x_train]

		idx_words_dev = [[[self.w2i[word.lower()] if word.lower() in self.w2i else self.w2i["<UNK>"] for word in sent]
							for sent in sample] 
							for sample in raw_x_dev]

		# index characters: sample -> sent -> word -> idx_char
		idx_chars_dev = [[[[self.c2i[char] if char in self.c2i else self.c2i["<UNK>"] for char in word] for word in sent]
							for sent in sample] 
							for sample in raw_x_dev]							

		idx_words_test = [[[self.w2i[word.lower()] if word.lower() in self.w2i else self.w2i["<UNK>"] for word in sent]
							for sent in sample] 
							for sample in raw_x_test]

		# index characters: sample -> sent -> word -> idx_char
		idx_chars_test = [[[[self.c2i[char] if char in self.c2i else self.c2i["<UNK>"] for char in word] for word in sent]
							for sent in sample] 
							for sample in raw_x_test]							

		nb_samples = len(raw_x_train)

		saver = tf.train.Saver()

		max_acc = 0

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			if "load_path" in self.params and self.params["load_path"] != "":
				print("Model is loaded from {}".format(self.params["load_path"]))
				saver.restore(sess, self.params["load_path"])

			print("Training steps:")					

			for epoch in range(self.nb_epochs):
				random_idx = np.random.permutation(nb_samples)

				# shuffle data before each epoch
				shuffled_idx_words, shuffled_idx_chars, shuffled_y = [], [], []
				for i in range(nb_samples):
					shuffled_idx_words.append(idx_words_train[random_idx[i]])
					shuffled_idx_chars.append(idx_chars_train[random_idx[i]])
					shuffled_y.append(raw_y_train[random_idx[i]])

				# go through the entire shuffled dataset, take batches to feed into the network
				current_idx = 0
				avg_loss = []
				avg_acc = []
				while current_idx < nb_samples:
					batch_words, real_length_sents, batch_chars, batch_labels, current_idx = self.get_batch(shuffled_idx_words, shuffled_idx_chars, shuffled_y, current_idx)
					if batch_labels.shape[0] < self.batch_size:
						break

					loss1, acc1, _ = sess.run([self.loss, self.acc, self.opt], feed_dict={self.tf_word_ids: batch_words,
					 			 										  self.tf_sentence_lengths: real_length_sents,
					 			 										  self.tf_target_matrix: batch_labels,
					 			 										  self.tf_keep_prob: self.keep_prob,
					 			 										  self.tf_learning_rate: 0.001,
					 			 										  self.tf_char_ids: batch_chars,
					 			 										  self.tf_doc_boundaries: self.doc_boundaries})
					avg_loss.append(loss1)
					avg_acc.append(acc1)

				mean_loss = np.mean(avg_loss)
				mean_acc = np.mean(avg_acc)

				# evaluate the model performance on the dev. set
				acc_dev, _ = self.accuracy(sess, idx_words_dev, idx_chars_dev, raw_y_dev)
				acc_train, _ = self.accuracy(sess, idx_words_train, idx_chars_train, raw_y_train)
				acc_test, _ = self.accuracy(sess, idx_words_test, idx_chars_test, raw_y_test)

				print("Epoch {} | training set: loss {}, acc {} | dev. set: acc {} | test. set: acc {}".format(epoch, mean_loss, acc_train, acc_dev, acc_test))

				# save the model				
				# if max_acc < acc_train:
				if max_acc < acc_dev:
					saver.save(sess, self.params["save_path"])	
					max_acc = acc_train
					print("Model is saved.")

			# print("acc on the test set:", self.accuracy(sess, idx_words_test, idx_chars_test, raw_y_test))
		

	def accuracy(self, sess, idx_words, idx_chars, raw_y, return_pred=False):		
		nb_samples = len(raw_y)

		outputs = []
		real_lengths = []
		labels = []

		current_idx = 0
		while current_idx < nb_samples:
			batch_words, real_length_sents, batch_chars, batch_labels, current_idx = self.get_batch(idx_words, idx_chars, raw_y, current_idx)
			if batch_labels.shape[0] < self.batch_size:
				break

			real_lengths.append(self.doc_boundaries)
			labels.append(batch_labels)

			out = sess.run(self.pred, feed_dict={self.tf_word_ids: batch_words,
						 					self.tf_sentence_lengths: real_length_sents,
						 					self.tf_keep_prob: 1,
											self.tf_doc_boundaries: self.doc_boundaries,
											self.tf_char_ids: batch_chars})

			real_out = []
			for o, m, l in zip(out, batch_labels, self.doc_boundaries):
				# real_out.append([o, m])
				real_out.append([o[:l, :l], m[:l, :l]])

			outputs +=real_out

		# calculate accuracy using a particular threshold
		accuracies = []
		for a, b in outputs:				
			d = (a == b) + 0
			e = np.sum(d)
			f = np.size(d)
			g = e/f
			accuracies.append(g)

		avg_acc = np.mean(accuracies)

		if return_pred:
			return avg_acc, outputs
		else:
			return avg_acc, None		

	def get_batch(self, words, chars, labels, start_idx):
		nb_samples = len(words)
		end_idx = start_idx + self.batch_size
		if end_idx > nb_samples:
			end_idx = nb_samples

		# sample -> sent -> word
		batch_words = words[start_idx: end_idx]

		# sample -> sent -> word -> char		
		batch_chars = chars[start_idx: end_idx]

		# list of matrices [nb_sents, nb_sents]
		batch_labels = labels[start_idx: end_idx]


		# pad sentences
		# find the number of words in the longest sentence of batch word, and store real length of sentences
		nb_words_in_max_sent = 0
		real_length_sents = []
		for sample in batch_words:
			real_length_sents += [len(sent) for sent in sample]
			for sent in sample:				
				if nb_words_in_max_sent < len(sent):
					nb_words_in_max_sent = len(sent)		

		real_length_sents = np.array(real_length_sents)
		
		# pad sentences in order to all sentences in the batch have the same length
		for i in range(len(batch_words)):
			for j in range(len(batch_words[i])):
				if len(batch_words[i][j]) < nb_words_in_max_sent:
					batch_words[i][j] = np.lib.pad(batch_words[i][j], (0, nb_words_in_max_sent - len(batch_words[i][j])), 'constant',
						constant_values=(self.w2i["<PAD>"], self.w2i["<PAD>"]))
		
		# remove document boundaries: X[nb_doc, [nb_sents], nb_words] -> X[new_nb_sents, nb_words]
		# where new_nb_sents = sum(nb_sents) of all documents in the batch
		new_batch_words = []
		# keep document boundaries in order to re-split documents from the batch of sentences
		self.doc_boundaries = []

		for sample in batch_words:
			self.doc_boundaries.append(len(sample))
			new_batch_words += [sent for sent in sample]			

		# pad chars
		nb_char_in_longest_word = 0
		new_batch_chars = []
		for sample in batch_chars:
			new_batch_chars += [sent for sent in sample]
			for sent in sample:
				if len(sent) == 0:
					continue
				longest_word = max([len(word) for word in sent])
				if longest_word > nb_char_in_longest_word:
					nb_char_in_longest_word = longest_word		

		for s in range(len(new_batch_chars)):
			nb_words = len(new_batch_chars[s])
			for w in range(nb_words):
				if len(new_batch_chars[s][w]) < nb_char_in_longest_word:
					new_batch_chars[s][w] += [self.c2i["<PAD>"]]*(nb_char_in_longest_word - len(new_batch_chars[s][w]))
			new_batch_chars[s] += [[self.c2i["<PAD>"]]*nb_char_in_longest_word]*(nb_words_in_max_sent-len(new_batch_chars[s]))

		# pad labels
		# batch_labels: list of numpy array [nb_sents, nb_sents], sizes of these matrices are not the same
		# -> pad to [nb_samples, nb_sents, nb_sents]
		size_of_largest_matrix = max([m.shape[0] for m in batch_labels])
		
		for i in range(len(batch_labels)):
			size = batch_labels[i].shape[0]
			batch_labels[i] = np.pad(batch_labels[i], ((0, size_of_largest_matrix - size), (0, size_of_largest_matrix - size)),
				'constant', constant_values=(0, 0))

		new_batch_words = np.array(new_batch_words)
		new_batch_chars = np.array(new_batch_chars)
		new_batch_labels = np.array(batch_labels)		

		return new_batch_words, real_length_sents, new_batch_chars, new_batch_labels, end_idx

	# helper functions
	def load_word_emb(self, emb_file):
		model = KeyedVectors.load_word2vec_format(emb_file, binary=False)
		loaded_words = 0
		for word in self.w2i:
			if word in model:
				self.word_emb[self.w2i[word]] = model[word]
				loaded_words += 1

		print("There was {} words loaded from Glove.".format(loaded_words))

	def create_dicts(self, train_pkl_file, dev_pkl_file, output_file=None):
		dataset = pickle.load(open(train_pkl_file, "rb"))[0]
		dataset += pickle.load(open(dev_pkl_file, "rb"))[0]

		# create word and char dictionaries
		w2i, i2w, c2i, i2c = {"<UNK>": 0, "<PAD>": 1}, {0: "<UNK>", 1: "<PAD>"}, {"<UNK>": 0, "<PAD>": 1}, {0: "<UNK>", 1: "<PAD>"}

		# each sample consists of document and binary matrix, where document is a set of sentences
		for sample in dataset:
			for sent in sample:
				for word in sent:
					for char in word:
							if char not in c2i:
								c2i[char] = len(c2i)
								i2c[len(i2c)] = char

					l_word = word.lower()
					if l_word not in w2i:
						w2i[l_word] = len(w2i)
						i2w[len(i2w)] = l_word					

		if output_file is not None:
			pickle.dump({"w2i": w2i, "i2w": i2w, "c2i": c2i, "i2c": i2c}, open(file=output_file, mode="wb"))		

		print(c2i)

	def test(self, model_path, test_file, output_folder=""):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print("Restore the best saved model.")
			saver = tf.train.Saver()
			saver.restore(sess, model_path)

			[raw_x_test, raw_y_test] = pickle.load(open(test_file, "rb"))

			# index words: sample -> sent -> idx_word
			idx_words_test = [[[self.w2i[word.lower()] if word.lower() in self.w2i else self.w2i["<UNK>"] for word in sent]
								for sent in sample] 
								for sample in raw_x_test]

			# index characters: sample -> sent -> word -> idx_char
			idx_chars_test = [[[[self.c2i[char] if char in self.c2i else self.c2i["<UNK>"] for char in word] for word in sent]
								for sent in sample] 
								for sample in raw_x_test]			

			acc, outputs = self.accuracy(sess, idx_words_test, idx_chars_test, raw_y_test, return_pred=output_folder!="")
			print("acc on the {}: {}".format(test_file, acc))
			
			# save outputs to file
			if output_folder!="":
				for i in range(len(outputs)):
					np.savetxt("{}/pred{}.txt".format(output_folder, i), outputs[i][0], fmt="%d")					
					np.savetxt("{}/label{}.txt".format(output_folder, i), outputs[i][1], fmt="%d")
					np.savetxt("{}/diff{}.txt".format(output_folder, i), outputs[i][1]-outputs[i][0], fmt="%2d")

					
# params = {"dicts_file": "dict.pkl",
# 		  "word_dim": 100, 
# 		  "word_emb": "../../Documents/ner/pretrained_embeddings/glove.6B.100d.txt",
# 		  "char_dim": 32,
# 		  "conv1": [2, 32],
# 		  "conv2": [3, 64],
# 		  "conv3": [4, 64],
# 		  "lstm_num_units": 128,
# 		  "keep_prob": 0.5,
# 		  "batch_size": 3,
# 		  "nb_epochs": 40,
# 		  # "load_path":"./model/ontonotes",
# 		  "save_path": "./model/ontonotes"}

# model = sent_model(params)

# # model.train("train.pkl", "dev.pkl", "test.pkl")
# model.train("train.pkl", "dev.pkl", "test.pkl")
# # model.test("./model/ontonotes", "test.pkl")

params = {"dicts_file": "qbcoref_dict.pkl",
		  "word_dim": 100, 
		  # "word_emb": "../../Documents/ner/pretrained_embeddings/glove.6B.100d.txt",
		  "char_dim": 64,
		  "conv1": [2, 64],
		  "conv2": [3, 128],
		  "conv3": [4, 128],
		  "lstm_num_units": 128,
		  "keep_prob": 0.5,
		  "batch_size": 5,
		  "nb_epochs": 50,
		  # "load_path":"./model/qbcoref",
		  "save_path": "./model/qbcoref"}

model = sent_model(params)

# model.train("train.pkl", "dev.pkl", "test.pkl")
model.train("qbcoref_train.pkl", "qbcoref_dev.pkl", "qbcoref_test.pkl")
# model.test("./model/qbcoref", "test.pkl")
# model.test("./model/ontonotes", "qbcoref_train.pkl", "out_qbcoref_train")

