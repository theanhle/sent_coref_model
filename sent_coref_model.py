import numpy as np 
import tensorflow as tf 
import pickle
from gensim.models import KeyedVectors
import string
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d


class sent_model:
	def __init__(self, params=None):
		self.log_dis = {}
		self.log_dis[0] = [0, 0]
		for i in range(1, 10):
			self.log_dis[i] = [self.log_dis[i-1][1] + 1, self.log_dis[i-1][1] + 2**i]		

		self.params = params

		self.keep_prob = params["keep_prob"]
		self.nb_epochs = params["nb_epochs"]

		# load word and char dictionaries
		dicts = pickle.load(open(params["dicts_file"], "rb"))
		self.w2i = dicts["w2i"]
		self.i2w = dicts["i2w"]
		self.c2i = dicts["c2i"]
		self.i2c = dicts["i2c"]		

		self.word_dim = params["word_dim"]
		self.word_vocab_size = len(self.w2i)
		self.char_dim = params["char_dim"]
		self.char_vocab_size = len(self.c2i)

		print("Sizes of word and char dictionaries: {}, {}".format(self.word_vocab_size, self.char_vocab_size))	

		self.word_emb = np.zeros(shape=(self.word_vocab_size, self.word_dim))		
		# load word embedding		
		if "word_emb" in params:
			print("pre-trained word embedding {} is being loaded ...".format(params["word_emb"]))
			self.load_word_emb(params["word_emb"])

		tf.reset_default_graph()
		
		# [sent, word]
		self.tf_word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="word_ids")
		# real length sents
		self.tf_sentence_lengths= tf.placeholder(dtype=tf.int32, shape=[None], name="sentence_lengths")

		# [sent, word, char]
		self.tf_char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="char_ids")

		# binary matrix representing the relationship between sents: [sent, sent]		
		self.tf_target_matrix = tf.placeholder(dtype=tf.int32, shape=[None, None], name="target_matrix")

		# keep_prob
		self.tf_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")

		# learning rate
		self.tf_learning_rate= tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

		# distant sentences
		self.tf_distant_sents = tf.placeholder(dtype=tf.int32, shape=[None, None], name="tf_distant_sents")
		
		# load word embedding
		with tf.variable_scope("word_embedding"):
			tf_word_embeddings = tf.Variable(self.word_emb, dtype=tf.float32,
				trainable=True, name="word_embedding")
			embedded_words = tf.nn.embedding_lookup(tf_word_embeddings, self.tf_word_ids, name="embedded_words")
			self.input = embedded_words # sent, word, word_dim


		# CNN network to capture character-level features
		with tf.variable_scope("char_cnn"):
			tf_char_embeddings = tf.get_variable(name="char_embeddings",
												 dtype=tf.float32,
												 shape=[self.char_vocab_size, self.char_dim],
												 trainable=True,
												 initializer=xavier_initializer())

			conv = tf.nn.embedding_lookup(tf_char_embeddings,
										  self.tf_char_ids,
										  name="embedded_cnn_chars")
			for i, (ks, fil) in enumerate(self.params["conv"]):
				conv = tf.layers.conv2d(inputs=conv, # sent, word, char, feature
										filters=fil,
										kernel_size=(1, ks),
										strides=(1, 1),
										padding="same",
										name="conv_{}".format(i),
										kernel_initializer=xavier_initializer_conv2d())
			# sent, word, char, cnn_feature
			# shape = tf.shape(conv)
			# conv = tf.nn.dropout(x=conv, keep_prob=self.tf_keep_prob, noise_shape=[shape[0], 1, shape[2], shape[3]])
			# conv = tf.nn.dropout(x=conv, keep_prob=self.tf_keep_prob)
			self.char_cnn = tf.reduce_max(conv, axis=2) # sent, word, cnn_feature
			self.input = tf.nn.dropout(tf.concat([self.input, self.char_cnn], axis=-1), self.tf_keep_prob) # [sents, words, word_dim + cnn_features]

		# Bi-LSTM to generate final input representation in combination with both left and right contexts
		with tf.variable_scope("bi_lstm_words"):
			cell_fw = tf.contrib.rnn.LSTMCell(self.params["word_lstm_units"])
			cell_bw = tf.contrib.rnn.LSTMCell(self.params["word_lstm_units"])
			(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.input,
																		sequence_length=self.tf_sentence_lengths,
																		dtype=tf.float32)			

			bilstm_output = tf.concat([output_fw, output_bw], axis=-1) # [sents, words, 2*lstm_units]

			mask = tf.where(condition=tf.equal(self.tf_word_ids, self.w2i["<PAD>"]), x=-1e10*tf.ones_like(self.tf_word_ids, dtype=tf.float32), y=tf.zeros_like(self.tf_word_ids, dtype=tf.float32))
			mask = tf.tile(tf.expand_dims(mask, -1), (1, 1, 2*self.params["word_lstm_units"]))

			bilstm_output = bilstm_output + mask
			# bilstm_output = tf.nn.dropout(tf.reduce_max(bilstm_output, axis=1), self.tf_keep_prob) # [sents, 2*word_lstm_units]
			bilstm_output = tf.reduce_max(bilstm_output, axis=1) # [sents, 2*word_lstm_units]

		# represent sents in their contexts 
		with tf.variable_scope("sent_representation"):			
			# sent_rep = tf.layers.conv1d(inputs=sent_rep[None, :, :], filters=2*self.lstm_num_units, kernel_size=3, padding='same')			
			# sent_rep = tf.squeeze(sent_rep, axis=0)			
			sent_fw = tf.contrib.rnn.LSTMCell(self.params["sent_lstm_units"])
			sent_bw = tf.contrib.rnn.LSTMCell(self.params["sent_lstm_units"])
			(output_sent_fw, output_sent_bw), _ = tf.nn.bidirectional_dynamic_rnn(sent_fw, sent_bw, bilstm_output[None, :, :],
																		# sequence_length=self.tf_sentence_lengths,
																		dtype=tf.float32)
			sent_bilstm = tf.concat([output_sent_fw, output_sent_bw], axis=-1)

			sent_rep = tf.squeeze(sent_bilstm, axis=0) # [sents, 2*sent_lstm_units]

		with tf.variable_scope("self_att"):
			distant_embeddings = tf.get_variable(name="distant_embeddings",
												 dtype=tf.float32,
												 shape=[10, 30],
												 trainable=True,
												 initializer=xavier_initializer()
												 )
			embedded_disant_sent = tf.nn.embedding_lookup(distant_embeddings, self.tf_distant_sents, name="embedded_disant_sent") # [sent, sent, 20]

			x = sent_rep
			de = x.get_shape()[-1]
			w = tf.get_variable(dtype=tf.float32, shape=[de], trainable=True, name="w")
			w1 = tf.get_variable(dtype=tf.float32, shape=[de, de], trainable=True, name="W1")
			w2 = tf.get_variable(dtype=tf.float32, shape=[de, de], trainable=True, name="W2")
			b1 = tf.get_variable(dtype=tf.float32, shape=[de], trainable=True, name="b1")
			b = tf.get_variable(dtype=tf.float32, shape=[], trainable=True, name="b")

			x_w1 = tf.matmul(x, w1) # n, de
			s = tf.map_fn(lambda xj: x_w1 + tf.tensordot(w2, xj, axes=1) + b1, x) # n, n, de
			s = s + tf.layers.dense(embedded_disant_sent, de, use_bias=False, reuse=tf.AUTO_REUSE)
			f = tf.tensordot(tf.tanh(s), w, axes=1) + b # n, n
			weight = tf.nn.softmax(f, axis=1) # n, n
			h = tf.transpose(tf.map_fn(lambda weight_j: tf.transpose(x)*weight_j, weight), [0, 2, 1]) # [sents, sents, 4*sent_lstm_units]
						
		with tf.variable_scope("loss_and_opt"):
			self.logits = tf.nn.dropout(tf.layers.dense(inputs=h, units=2, activation=None), self.tf_keep_prob)

			self.logits = 0.5*(self.logits + tf.transpose(self.logits, [1, 0, 2]))

			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.tf_target_matrix, 2),
														   logits=self.logits,
														   name="loss_function"))

			self.pred = tf.argmax(self.logits, axis=-1)
			eq = tf.cast(tf.equal(tf.cast(self.pred, tf.int32), self.tf_target_matrix), tf.float32)
			self.acc = tf.reduce_mean(eq)

			self.opt = tf.train.AdamOptimizer(learning_rate=self.tf_learning_rate).minimize(self.loss)
		
		
	def train(self, train_pkl_file, dev_pkl_file, test_pkl_file):
		[raw_x_train, raw_y_train] = pickle.load(open(train_pkl_file, "rb"))
		[raw_x_dev, raw_y_dev] = pickle.load(open(dev_pkl_file, "rb"))
		[raw_x_test, raw_y_test] = pickle.load(open(test_pkl_file, "rb"))

		# index words: sample -> sent -> idx_word
		idx_words_train = [[[self.w2i[word.lower()] if word.lower() in self.w2i else self.w2i["<UNK>"] for word in sent if word!=""]
							for sent in sample] 
							for sample in raw_x_train]

		# index characters: sample -> sent -> word -> idx_char
		idx_chars_train = [[[[self.c2i[char] if char in self.c2i else self.c2i["<UNK>"] for char in word] for word in sent if word!=""]
							for sent in sample] 
							for sample in raw_x_train]

		idx_words_dev = [[[self.w2i[word.lower()] if word.lower() in self.w2i else self.w2i["<UNK>"] for word in sent if word!=""]
							for sent in sample] 
							for sample in raw_x_dev]

		# index characters: sample -> sent -> word -> idx_char
		idx_chars_dev = [[[[self.c2i[char] if char in self.c2i else self.c2i["<UNK>"] for char in word] for word in sent if word!=""]
							for sent in sample] 
							for sample in raw_x_dev]							

		idx_words_test = [[[self.w2i[word.lower()] if word.lower() in self.w2i else self.w2i["<UNK>"] for word in sent if word!=""]
							for sent in sample] 
							for sample in raw_x_test]

		# index characters: sample -> sent -> word -> idx_char
		idx_chars_test = [[[[self.c2i[char] if char in self.c2i else self.c2i["<UNK>"] for char in word] for word in sent if word!=""]
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
				max_acc = self.accuracy(sess, idx_words_dev, idx_chars_dev, raw_y_dev)				
				print("The best acc on the dev. set:", max_acc)

			print("Model params:", self.params)					

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
				while current_idx < nb_samples:
					# batch_words, real_length_sents, batch_chars, label, current_idx = self.get_batch(shuffled_idx_words, shuffled_idx_chars, shuffled_y, current_idx)					
					batch_words, real_length_sents, batch_chars, label, current_idx, distant_sents = self.get_batch(shuffled_idx_words, shuffled_idx_chars, shuffled_y, current_idx)					

					loss1, _ = sess.run([self.loss, self.opt], feed_dict={self.tf_word_ids: batch_words,
					 			 										  self.tf_target_matrix: label,
					 			 										  self.tf_keep_prob: self.keep_prob,
					 			 										  self.tf_char_ids: batch_chars,
					 			 										  self.tf_learning_rate: 0.001,
					 			 										  self.tf_sentence_lengths: real_length_sents,
					 			 										  self.tf_distant_sents: distant_sents
					 			 										  })
					avg_loss.append(loss1)

				mean_loss = np.mean(avg_loss)

				acc_dev = self.accuracy(sess, idx_words_dev, idx_chars_dev, raw_y_dev)
				acc_train = self.accuracy(sess, idx_words_train, idx_chars_train, raw_y_train)
				acc_test = self.accuracy(sess, idx_words_test, idx_chars_test, raw_y_test)

				if max_acc < acc_dev:
					if "save_path" in self.params:
						saver.save(sess, self.params["save_path"])
					max_acc = acc_dev
					print("Epoch {:2d} | Loss {:.4f} | Acc on the training set: {:.4f}, dev. set: {:.4f}, test. set: {:.4f} (*)".format(epoch, mean_loss, acc_train, acc_dev, acc_test))
				else:
					print("Epoch {:2d} | Loss {:.4f} | Acc on the training set: {:.4f}, dev. set: {:.4f}, test. set: {:.4f}".format(epoch, mean_loss, acc_train, acc_dev, acc_test))


	def accuracy(self, sess, idx_words, idx_chars, raw_y, output_file=""):
		nb_samples = len(raw_y)

		outputs = []

		current_idx = 0
		while current_idx < nb_samples:
			# batch_words, real_length_sents, batch_chars, label, current_idx = self.get_batch(idx_words, idx_chars, raw_y, current_idx)
			batch_words, real_length_sents, batch_chars, label, current_idx, distant_sents = self.get_batch(idx_words, idx_chars, raw_y, current_idx)

			out = sess.run(self.pred, feed_dict={self.tf_word_ids: batch_words,
						 					self.tf_sentence_lengths: real_length_sents,
						 					self.tf_keep_prob: 1,
											self.tf_char_ids: batch_chars,
											self.tf_distant_sents: distant_sents
											})
			outputs.append([label, out])

		accuracies = []
		for a, b in outputs:		
			assert(a.shape==b.shape)
			# d = (a == b) + 0
			# e = np.sum(d)
			# f = np.size(d)
			# g = e/f

			n = a.shape[0]
			nb_corrects = 0
			for i in range(n):
			    for j in range(i, n):
			        nb_corrects += int(a[i, j]==b[i, j])

			# accuracy = nb_corrects/(n*(n+1)/2)
			g = 2*nb_corrects/n/(n+1)
			accuracies.append(g)

		avg_acc = np.mean(accuracies)				
		if output_file == "":
			return avg_acc
		else:
			return avg_acc, outputs

	def get_batch(self, words, chars, labels, idx):
		# words: doc, sent, indexed_word
		# chars: doc, sent, word, indexed_char
		# output words, chars, labels of one document

		batch_words = [[word for word in sent] for sent in words[idx]] # [sent, word]

		# index characters: document -> sent -> word -> idx_char
		batch_chars = [[[char for char in word] for word in sent] for sent in chars[idx]] # [sent, word, char]

		label = labels[idx] # [sent, sent]

		# pad sentences
		# find the number of words in the longest sentence of batch word, and store real length of sentences
		real_length_sents = np.array([len(sent) for sent in batch_words])
		nb_words_in_max_sent = max(real_length_sents)		
		
		# pad sentences in order to all sentences in the batch have the same length
		for j in range(len(batch_words)):
			if len(batch_words[j]) < nb_words_in_max_sent:
				batch_words[j] = np.lib.pad(batch_words[j], (0, nb_words_in_max_sent - len(batch_words[j])), 'constant',
					constant_values=(self.w2i["<PAD>"], self.w2i["<PAD>"]))
		# pad chars
		nb_chars_in_longest_word = 0
		for sent in batch_chars:
			max_len = max([len(word) for word in sent])
			if max_len > nb_chars_in_longest_word:
				nb_chars_in_longest_word = max_len

		for i in range(len(batch_chars)):
			for j in range(len(batch_chars[i])):
				if len(batch_chars[i][j]) < nb_chars_in_longest_word:
					batch_chars[i][j] +=[self.c2i["<PAD>"]]*(nb_chars_in_longest_word - len(batch_chars[i][j]))
			if real_length_sents[i] < nb_words_in_max_sent:
				batch_chars[i] += [[self.c2i["<PAD>"]]*nb_chars_in_longest_word]*(nb_words_in_max_sent - real_length_sents[i])		

		batch_words = np.array(batch_words)
		batch_chars = np.array(batch_chars)
		label = np.array(label)				

		# distant_sents = [i//20 for i in np.arange(batch_words.shape[0])]		
		
		distant_sents = self.create_log_distances(batch_words.shape[0])

		return batch_words, real_length_sents, batch_chars, label, (idx + 1), distant_sents

	def create_log_distances(self, nb_sents):
		distances = np.zeros((nb_sents, nb_sents))
		for i in range(nb_sents):
			for j in range(nb_sents):
				distances[i, j] = abs(i-j)			

		def distance_to_log_dis(distance):
			for key in self.log_dis:
				if self.log_dis[key][0] <= distance <= self.log_dis[key][1]:
					return key
		return np.array([[distance_to_log_dis(i) for i in row] for row in distances])

	# helper functions
	def load_word_emb(self, emb_file):
		model = KeyedVectors.load_word2vec_format(emb_file, binary=False)
		loaded_words = 0
		for word in self.w2i:
			if word in model:
				self.word_emb[self.w2i[word]] = model[word]
				loaded_words += 1

		print("There are {} words loaded from Glove.".format(loaded_words))

	@staticmethod
	def create_dicts(train_pkl_file, dev_pkl_file, output_file=None):
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

	@staticmethod
	def sent_coref_dataset_statistic():
		x, y = pickle.load(open("train.pkl", "rb"))
		longest_doc = max(len(doc) for doc in x)
		print(longest_doc)

	def test(self, model_path, test_file, output_file=""):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print("Restore the best saved model.")
			saver = tf.train.Saver()
			saver.restore(sess, model_path)

			[raw_x_test, raw_y_test] = pickle.load(open(test_file, "rb"))

			# index words: sample -> sent -> idx_word
			idx_words_test = [[[self.w2i[word.lower()] if word.lower() in self.w2i else self.w2i["<UNK>"] for word in sent if word!=""]
								for sent in sample] 
								for sample in raw_x_test]

			# index characters: sample -> sent -> word -> idx_char
			idx_chars_test = [[[[self.c2i[char] if char in self.c2i else self.c2i["<UNK>"] for char in word] for word in sent if word!=""]
								for sent in sample] 
								for sample in raw_x_test]			

			if output_file =="":
				acc = self.accuracy(sess, idx_words_test, idx_chars_test, raw_y_test)
			else:
				acc, outputs = self.accuracy(sess, idx_words_test, idx_chars_test, raw_y_test, output_file)

				fo = open(file=output_file, encoding="utf8", mode="w")
				fo.write("Acc on {}: {}".format(output_file, acc))
				for a, b in outputs:
					fo.write("\n")					
					n = a.shape[0]
					for i in range(n):
						for j in range(n):
							fo.write("{:2d}".format(a[i, j]))

						fo.write(" | ")

						for j in range(n):
							fo.write("{:2d}".format(b[i, j]))

						fo.write(" | ")

						for j in range(n):
							fo.write("{:2d}".format(abs(a[i, j] - b[i, j])))

						fo.write("\n")			
				fo.close()

			print("acc on the {}: {:.4f}".format(test_file, acc))			

					
params = {"dicts_file": "dict.pkl",
		  "word_dim": 100, 
		  "word_emb": "glove.6B.100d.txt",
		  "char_dim": 32,
		  "conv": [[3, 32], [5, 64]],
		  "word_lstm_units": 128,
		  "sent_lstm_units": 128,
		  "keep_prob": 0.5,
		  "nb_epochs": 60,
		  # "load_path":"./models/ontonotes/ontonotes",
		  # "save_path": "./models/ontonotes/ontonotes"
		  }

model = sent_model(params)
model.train("train.pkl", "dev.pkl", "test.pkl")
# model.test("./models/ontonotes/ontonotes", "test.pkl", "ontonotes_test.txt")

# sent_model.create_dicts("train.pkl", "dev.pkl", "dict.pkl")
# sent_model.sent_coref_dataset_statistic()