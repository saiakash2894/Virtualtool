 # -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib

import random
import numpy as np
import scipy as sp

from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.palettes import inferno
from bokeh.models import HoverTool, ColumnDataSource, BoxZoomTool, TapTool, Circle, ZoomInTool, ZoomOutTool, ResetTool, CustomJS, OpenURL, Div
from bokeh.models.widgets import Slider, TextInput, Button, RadioButtonGroup, DataTable, TableColumn, TextAreaInput,HTMLTemplateFormatter
from bokeh.layouts import column, row, widgetbox, gridplot
#from bokeh.events import DoubleTap
import copy
import math

import json
import pickle

# added new imports
# import RAKE

# import operator

# import pytextrank

import yake


from textblob import TextBlob as tb
from bokeh.events import Tap, SelectionGeometry, Press, MouseEnter, DoubleTap
from gensim.summarization.summarizer import summarize
import networkx as nx
import pandas as pd
import nltk
import collections
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import Counter,defaultdict
from heapq import nlargest
import re
import string

######## PAPER HANDLING LOGIC
texts = []
titles = []
years = []
authors = []
dois = []
texts_globals = []
titles_globals = []
years_globals = []
authors_globals = []
dois_globals = []
def load_papers(year_start, year_end):
	for paper in get_proceedings(year_start, year_end):
		txt = paper.clean_text.lower()
		texts.append(txt)
		titles.append(paper.title)
		years.append(paper.year)

	with open('texts.json', 'w') as outfile:
		json.dump(texts, outfile)
	with open('titles.json', 'w') as outfile:
		json.dump(titles, outfile)
	with open('years.json', 'w') as outfile:
		json.dump(years, outfile)

def open_papers(year_start, year_end):
	global texts, titles, years, texts_globals, titles_globals, years_globals, dois_globals, authors_globals, authors
	texts_ = []
	titles_ = []
	years_ = []
	authors_ = []
	dois_ = []

	with open('texts.json', 'r') as outfile:
		texts_ = json.load(outfile)
	with open('titles.json', 'r') as outfile:
		titles_ = json.load(outfile)
	with open('years.json', 'r') as outfile:
		years_ = json.load(outfile)
	with open('authors.json', 'r') as outfile:
		authors_ = json.load(outfile)
	with open('dois.json', 'r') as outfile:
		dois_ = json.load(outfile)	

	for idx, y in enumerate(years_):
		if y >= year_start and y <= year_end:
			texts.append(texts_[idx])
			titles.append(titles_[idx])
			years.append(y)
			authors.append(authors_[idx])
			dois.append(dois_[idx])

	texts_globals = texts
	titles_globals = titles
	years_globals = years
	authors_globals = authors
	dois_globals = dois


######## TOP MODELLING STUFF
#Construct and execute topic modelling
#Option 0: NMF, Option 1: LDA
no_features = 1000
no_topics = 50

vectorizer_ = []
feature_names = []
t_model = []
model_W = []
model_H = []
vocab_ = []
def build_models(option):
	global vectorizer_, t_model, model_W, model_H, feature_names, vocab_
	
	if option == 0:
		vectorizer_ = TfidfVectorizer(max_df=0.95, min_df=.05, max_features=no_features, stop_words='english')
	else:
		vectorizer_ = CountVectorizer(max_df=0.95, min_df=.05, max_features=no_features, stop_words='english')

	vec_fit_ = vectorizer_.fit_transform(texts)
	feature_names = vectorizer_.get_feature_names()
	vocab_ = vectorizer_.vocabulary_

	if option == 0:
		t_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(vec_fit_)
	else:
		t_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(vec_fit_)

	model_W = t_model.transform(vec_fit_)
	model_H = t_model.components_

def save_models():
	global t_model, model_H, model_W, feature_names, vocab_
	joblib.dump(t_model, 't_model.json')
	joblib.dump(model_W, 'model_W.json')
	joblib.dump(model_H, 'model_H.json')
	joblib.dump(feature_names, 'feature_names.json')
	joblib.dump(vocab_, 'vec_vocab.json')

def load_models(option):
	global t_model, model_H, model_W, feature_names, vocab_
	t_model = joblib.load('t_model.json')
	model_W = joblib.load('model_W.json')
	model_H = joblib.load('model_H.json')
	feature_names = joblib.load('feature_names.json')
	with open('vec_vocab.json', 'rb') as infile:
		vocab_ = pickle.load(infile, encoding='latin1')

	if option == 0:
		vectorizer_ = TfidfVectorizer(max_df=0.95, min_df=.05, max_features=no_features, stop_words='english')
	else:
		vectorizer_ = CountVectorizer(max_df=0.95, min_df=.05, max_features=no_features, stop_words='english')

	# vec_fit_ = vectorizer_.fit_transform(texts)
	# feature_names = vectorizer_.get_feature_names()
	
#Compare new text against the model
comp_model_ = []
def comp_new_text(option, text):
	global vectorizer_, t_model, comp_model_, vocab_
	comp_model_ = []
	if option == 0:
		comp_vectorizer_ = TfidfVectorizer(vocabulary=vocab_)
	else:
		comp_vectorizer_ = CountVectorizer(vocabulary=vocab_)

	cv_ = comp_vectorizer_.fit_transform(text)
	# print(type(cv_))
	# # temp_list = cv_.nonzero()
	# # row = temp_list[0]
	# # column = temp_list[1]
	# # print(row)
	# # print(column)
	# value = cv_.data
	# column_index = cv_.indices
	# row_pointers = cv_.indptr
	# print("hey")
	# print (value)
	# print(column_index)
	# print(row_pointers)
	# # for i in cv_:
	# # 	temp_list.append(list(i.A[0]))
	# #print (temp_list)

	# print (vocab_.items())
	# for key in column_index:
	# 	for t,key1 in vocab_.items():
	# 		if key == key1:
	# 			key_word.append(t)


	# print ("Hello")
	# print (key_word)
	# print (feature_names)
	# print (vocab_)
	# print (cv_)
	comp_model_ = t_model.transform(cv_)
	# print (comp_model_)

#Grab top x words for any topic
def grab_topics(model, feature_names, no_top_words):
	topics_names = []
	for topic_idx, topic in enumerate(model.components_):
		topics_names.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
	return topics_names
                        
#Display top x words and top y representative paper titles per topic
def display_topics_and_docs(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print ("Hello world display")
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print (titles[doc_index])
        print ("")
        print ("*******")

#Assigns any paper to its top topic
def assign_to_top_topic(W):
	topic_pick = []
	for w in W:
		topic_pick.append(np.argsort(w)[-1])
		#print("topic pick", topic_pick)
	return topic_pick

#Calculates similarity between paper X and all other papers
def find_most_similar(W, W_):
	# print W_
	#top_similarities = []
	similarities = []
	for idx, w in enumerate(W):
		similarities.append(cosine_similarity(W_, [w])[0][0])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

	#top_similarities = np.argsort(similarities)[-5:]

	
	#print (top_similarities)
	#for t in top_similarities:
		#print (similarities[t])

	# print ("******")

	#for di in top_similarities:
		#print (titles[di])
		#print (W[di])
		#print (years[di])
		#print (dois[di])

	return similarities

######## PLOTTING LOGIC
#Rotate a point around another by a given angle
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

#Place topic names on plot
def add_topic_names():
	topic_names = grab_topics(t_model, feature_names, 5)
	steps = 360. / no_topics

	for idx, t in enumerate(topic_names):

		pos = rotate((0,0), (0,(1.5)), math.radians(((steps * idx) + (steps * idx+1))/2.0))
		plt.text(pos[0], pos[1], topic_names[idx])

#Scatter plot similarities
def plt_sims(sims, tops):
	col = np.arange(25)
	steps = 360. / no_topics
	x = []
	y = []
	t = []

	for idx, s in enumerate(sims):
		pos = rotate((0,0), (0,(1.0-s)), math.radians(random.uniform(steps * tops[idx], steps * (tops[idx]+1))))
		x.append(pos[0])
		y.append(pos[1])
		t.append(col[tops[idx]])
	
	plt.circle(x, y, c = t)

	add_topic_names()

	plt.show()

new_id = -1
#Paper selection logic
def callback(attr, old, new):
	
	global source
	# print(new['1d']['indices'][0])
	name = source.data['desc'][new['1d']['indices'][0]]
	print(name)

	global new_id, id_interests
	new_id = new['1d']['indices'][0]
	# print(new_id)

	global table_data, chosen_papers
	data = dict(years_=[years[new_id]], titles_=[titles[new_id]], dois_=["https://doi.org/" + dois[new_id]])

	table_data.stream(data)

	# topic_names = grab_topics(t_model, feature_names, 5)

	update_plot()

#Redraw visualisation plot
def update_plot(highlight=False):
	global new_id, source, id_interests, topic_names, search_ids, custom_text
	#print(model_W[new_id])


	sims = find_most_similar(model_W, [model_W[new_id]])
	#print(sims)
	tops = assign_to_top_topic(model_W)

	col = inferno(no_topics)
	# random.shuffle(col)
	steps = 360. / no_topics
	x_ = []
	y_ = []
	t_ = []
	o_ = []
	topic_ = []

	# print ("id_interests", id_interests, "Search_ids", search_ids)

	case = 0
	if id_interests == [-1,-1]:
		case = 1

	for idx, s in enumerate(sims):
		pos = rotate((0,0), (0,(1.0-s)), math.radians(random.uniform(steps * tops[idx], steps * (tops[idx]+1))))
		x_.append(pos[0])
		y_.append(pos[1])
		t_.append(col[tops[idx]])
		if case == 0:
			if idx > id_interests[0] and idx < id_interests[1]:
				if len(search_ids) > 0 and idx in search_ids:
					o_.append(0.7)
				elif len(search_ids) == 0:
					o_.append(0.7)
				else:
					o_.append(0.12)
			else:
				o_.append(0.12)
		else:
			if idx in search_ids:
				o_.append(0.7)
			else:
				o_.append(0.12)

		topic_.append(topic_names[tops[idx]])
		# print (topic_names[tops[idx]])

	if highlight:
		t_[-1] = "blue"
		o_[-1] = 0.8

	source.data = dict(
				x=x_,
				y=y_,
				t=t_,
				o=o_,
				desc=titles,
				topics=topic_,
				auth=authors,
				year=years,
				dois=dois_globals,
				texts3= texts
				)

#handle year filtering
id_interests=[-1, -1]
def update_years(attr, old, new):
	global year_from, year_to, years_globals, texts, titles, id_interests
	year_to.start = min(year_from.value, 2016)

	texts_ = []
	titles_ = []
	years_ = []

	print (year_from.value, year_to.value)

	id_interests[1] = 0

	# print (years_globals[0:28])

	for idx, y in enumerate(years_globals):
		if y >= year_from.value and y <= year_to.value:

			if idx > 0 and years_globals[idx-1] < year_from.value:
				id_interests[0] = idx
			if idx > id_interests[1]:
				id_interests[1] = idx

	#Current year weirdness hack
	if year_from.value == 1981:
		id_interests[0] = 0

	# print(id_interests)

#Handle custom topic-match text
def process_text():
	global custom_text, comp_model_, new_id, model_W, texts, titles, years
	#print("Hello world")
	comp_new_text(1, [custom_text.value])

	#print (comp_model_.shape)
	#print ([comp_model_])
	model_W = np.concatenate((model_W, np.asarray([comp_model_[0]])))
	#print (model_W.shape)
	texts.append(custom_text.value)
	titles.append("Your text")
	years.append(2019)
	authors.append(["You"])
	dois.append(-1)
	
	# print ("******")
	# print(model_W)
	new_id=len(texts)-1
	update_plot(True)
	#print (custom_text.value)


#Handle custom topic-match text
def process_text1():
	global custom_text1, comp_model_, new_id, model_W, texts, titles, years
	#print("Hello world")
	comp_new_text(1, [custom_text1.value])

	#print (comp_model_.shape)
	#print ([comp_model_])
	model_W = np.concatenate((model_W, np.asarray([comp_model_[0]])))
	#print (model_W.shape)
	texts.append(custom_text1.value)
	titles.append("Your text")
	years.append(2019)
	authors.append(["You"])
	dois.append(-1)
	
	# print ("******")
	# print(model_W)
	new_id=len(texts)-1
	update_plot(True)
	#print (custom_text.value)


#Handle custom search text
search_ids = []
def process_search():
	global radio_search, search_text, titles, search_ids, id_interests, search_items, texts, years, dois, new_id, authors_globals
	print (search_text.value)
	print (radio_search.active)

	s_text = str(search_text.value).lower().split(';')
	author_text = str(search_text.value)

	print (s_text)
	print (radio_search.labels[radio_search.active])

	search_ids = []

	flag_in = False
	flag_author = False
	if radio_search.active == 0:
		for idx, w in enumerate(titles):
			flag_in = True
			for ss in s_text:
				if ss not in w.lower():
					flag_in = False
			if flag_in:
				search_ids.append(idx)
	elif radio_search.active == 2:
		for idx, d in enumerate(dois):
			if s_text[0] in d:
				new_id = idx
	elif radio_search.active == 3:
		for element in author_text.split(","):
			whitelist = string.ascii_letters + string.digits + ' '
			new_s = ''.join(c for c in element if c in whitelist)
			print("element:" + new_s.strip())
			for i in range (len(authors_globals)):
				# print(authors_globals[i])
				for d in authors_globals[i]:
					# print(d)
					if d == new_s.strip():
						flag_author = True
						print(flag_author)
						print(d)
						search_ids.append(i)
					
					
	else:
		for idx, w in enumerate(texts):
			flag_in = True
			for ss in s_text:
				if ss not in w.lower():
					flag_in = False
			if flag_in:
				search_ids.append(idx)
					
	if flag_in:
		search_ids.append(idx)

	print("search id")
	print (search_ids)
	# id_interests = [-1,-1]

	update_plot()

	
	#chosen_papers.source.data.update(table_data)


	# search_items.labels.append(search_text.value)
	# search_items.active.append(1)
	# if search_items == []:
	# 	search_items = CheckboxButtonGroup(labels=["blah", "blah"], active=[0,1])

# #Update paper data table
def update_table():
	global dict1, top_sims, texts1, summary1, dict2,dict3,table_data, chosen_papers, table_data1, chosen_papers1, Keyword_text, keyword_data
	summary1 = []
	texts1 =[]
	top_sims1 =[]
	
		
	print("hey blah")
	sims = find_most_similar(model_W, [model_W[new_id]])

	print ("sims length", len(sims))

	top_sims = np.argsort(sims)[-7:]

	print ("no error here")

	print ("yaers length", len(years))
	print ("titles length", len(titles))
	print ("dois length", len(dois))

	texts2 = [texts[di] for di in top_sims[0:50]]
	dict3 = dict(keywords_ = keywords1(texts2))
	keyword_data =  ColumnDataSource(data = dict3)

	Keyword_text.source.data.update(keyword_data.data)
	# value4 =' , '.join(keywords1(texts2))
	# Keyword_text.value = value4
	

	# print(texts1)

	# dois.append(-1)
	# maximum = max(sims)
	# print (maximum)
	# for t in sims:
	# 	if len(top_sims1)<= 5:
	# 		if t <= maximum:
	# 			top_sims1.append(t)
	# # top_sims = np.argsort(sims)[0:5]
	# top_sims = np.argsort(top_sims1)

	# print(top_sims)
	

	print ("finished updating the table")
	for di in top_sims[0:5]:
		texts1 =[]
		texts1.append(texts[di])
		# print(texts1)
		summary1.append(sentences1(texts1,0))

	# for di in top_sims[0:5]:
	# 	dois[di] = ' http://dx.doi.org/'+ dois[di]
	# 	print(dois[di])
	# 	doi1.append(dois[di])

	dict1= dict(
					years_=[years[di] for di in top_sims[0:5]],
					titles_=[titles[di] for di in top_sims[0:5]],
					authors_=[authors[di] for di in top_sims[0:5]],
					dois_= [dois[di] for di in top_sims[0:5]],
					summary_ = summary1)
	# dict2= dict(
	# 				summary_ = summary1)

 
	table_data =  ColumnDataSource(data = dict1)

	chosen_papers.source.data.update(table_data.data)

	# table_data1 =  ColumnDataSource(data = dict2)

	# chosen_papers1.source.data.update(table_data1.data)

	

	
# def update_table1(attr,old,new):
# 	print ("Updating the table")
# 	pass
def sentences1(text2,option):

	print("no error")
	global sentences
	summarize_text = []
	sentences =[]
	print(len(text2))
	if option == 1:
		print("yes")
		sentences.append(sent_tokenize(str(text2)))
	else:

		for s in text2:
			sentences.append(sent_tokenize(s))

	sentences = [y for x in sentences for y in x] # flatten list
	# print(sentences[:5])
	# print(len(sentences))

	# Extract word vectors
	# word_embeddings = {}
	# f = open('glove.6B.100d.txt', encoding='utf-8')
	# for line in f:
	# 	values = line.split()
	# 	word = values[0]
	# 	coefs = np.asarray(values[1:], dtype='float32')
	# 	word_embeddings[word] = coefs
	# f.close()
	# print(len(word_embeddings))

	# remove punctuations, numbers and special characters
	clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

	# make alphabets lowercase
	clean_sentences = [str(s).lower() for s in clean_sentences]

	
	# remove stopwords from the sentences
	clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


	# # vectorising the sentences
	# sentence_vectors = []
	# for i in clean_sentences:
	# 	if len(i) != 0:
	# 		v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
	# 	else:
	# 		v = np.zeros((100,))
	# 	sentence_vectors.append(v)


	# # similarity matrix
	# print(len(sentences))
	# sim_mat = np.zeros([len(sentences), len(sentences)])

	# for i in range(len(sentences)):
	# 	for j in range(len(sentences)):
	# 		if i != j:
	# 			sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

	# print("reached")



	# # Applying page rank algorithm
	# nx_graph = nx.from_numpy_array(sim_mat)
	# scores = nx.pagerank(nx_graph)
	
	# ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
	# # Extract top 5 sentences as the summary
	# lis =[]
	# for i in range(5):
	# 	# lis.append(['figure','section'])
	# 	# digits = ['0','1','2','3','4','5','6','7','8','9']
	# 	# lis.append(digits)

	# 	# ranked_sentences_new = " ".join([j for j in ranked_sentences[i][1] if j not in lis])
	# 	print(ranked_sentences[i][1])

	# summarize_text.append(" ".join(ranked_sentence[i][1]))

	# print("Summarize Text: \n", ". ".join(summarize_text))


	#Method 2: Weighted Word Frequencies | Word Focused
	#Compute word frequencies for each sentence
	word_frequencies = {}
	for i in range(len(clean_sentences)):
		for word in nltk.word_tokenize(clean_sentences[i]):
			if word not in word_frequencies.keys():
				word_frequencies[word] = 1
			else:
				word_frequencies[word] += 1
	 
	#Find max frequency in text and compute the weighted frequency based on the maximum frequency.
	maximum_frequency = max(word_frequencies.values())
	for word in word_frequencies.keys():
		word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
	 
	#Apply scores to each UNCLEANED SENTENCE
	sentence_scores = {}
	for sent in sentences:
		for word in nltk.word_tokenize(sent.lower()):
			if word in word_frequencies.keys():
				if len(sent.split(' ')) < 30:
					if sent not in sentence_scores.keys():
						sentence_scores[sent] = word_frequencies[word]
					else:
						sentence_scores[sent] += word_frequencies[word]
	 
	#Choose number of sentences you want in your summary
	summary_sentences = nlargest(2, sentence_scores, key=sentence_scores.get)
	summary = ' '.join(summary_sentences)
	# print(summary)
	return summary




	# dict1= dict(
	# 				summary_=str(summary),
	# 				)

	# global table_data, chosen_papers
	# table_data =  ColumnDataSource(data = dict1)

	# chosen_papers.source.data.update(table_data.data)

	# print(summarize(text2.to_string()))


# function to remove stopwords
def remove_stopwords(sen):
	stop_words = stopwords.words('english')
	stop_words.append(['figure','section'])
	digits = ['0','1','2','3','4','5','6','7','8','9']
	stop_words.append(digits)

	sen_new = " ".join([i for i in sen if i not in stop_words])
	return sen_new





	
# 	text_string=','.join([str(i).lower() for i in text2])

# 	match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string) #return all the words with the number of characters in the range [3-15]

# 	fdist = nltk.FreqDist(match_pattern) # creates a frequency distribution  from a list
# 	most_common = fdist.max()    # returns a single element
# 	top_five = fdist.most_common(5)# returns a list

# 	list_5=[word for (word, freq) in fdist.most_common(5)]
# 	sent_tokenize_list = nltk.sent_tokenize(text_string)    
# 	for sentence in sent_tokenize_list:
# 		for word in list_5:
# 			if word in sentence:
# 				print(sentence)
# def tf(word, blob):
# 	return blob.words.count(word) / len(blob.words)

# def n_containing(word, bloblist):
# 	return sum(1 for blob in bloblist if word in blob)

# def idf(word, bloblist):
# 	return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

# def tfidf(word, blob, bloblist):
# 	return tf(word, blob) * idf(word, bloblist)
# def keywords1(texts_2):
# 	print("keywords")
# 	for blob in texts_2:
# 		print("Top words in document {}".format(i + 1))
# 		scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
# 		sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
# 		for word, score in sorted_words[:3]:
# 			print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
def keywords1(texts_2):
	global text4, keywords_yake
	keywords_yake = []
	# Reka setup with stopword directory
	text4 = str(texts_2)

	# # Using Rake
	# stop_dir = "SmartStoplist.txt"
	# rake_object = RAKE.Rake(stop_dir)
	# # Extract keywords
	# keywords = rake_object.run(text4)
	# print ("keywords: ", keywords[0:20])

	#Using Yake
	language = "en"
	max_ngram_size = 3
	deduplication_thresold = 0.9
	deduplication_algo = 'seqm'
	windowSize = 1
	numOfKeywords = 20

	custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
	keywords = custom_kw_extractor.extract_keywords(text4)

	for kw in keywords:
		print(kw)
		# print(kw[0])
		keywords_yake.append(kw[0])
	return keywords_yake
		

	# # Setup Term Extractor
	# extractor = extract.TermExtractor()
	# # Extract Keywords
	# keywords_topica = extractor(text)
	# print(keywords_topica)

	# Create dictionary to feed into json file

	# file_dic = {"id" : 0,"text" : text4}
	# file_dic = json.dumps(file_dic)
	# loaded_file_dic = json.loads(file_dic)

	# # Create test.json and feed file_dic into it.
	# with open('test.json', 'w') as outfile:
	# 	json.dump(loaded_file_dic, outfile)

	# path_stage0 = "test.json"
	# path_stage1 = "o1.json"

	# # Extract keyword using pytextrank
	# with open(path_stage1, 'w') as f:
	# 	for graf in pytextrank.parse_doc(pytextrank.json_iter(path_stage0)):
	# 		f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
	# 		print("I am here")
	# 		print(pytextrank.pretty_print(graf._asdict()))

	# path_stage1 = "o1.json"
	# path_stage2 = "o2.json"

	# graph, ranks = pytextrank.text_rank(path_stage1)
	# pytextrank.render_ranks(graph, ranks)

	# with open(path_stage2, 'w') as f:
	# 	for rl in pytextrank.normalize_key_phrases(path_stage1, ranks):
	# 		f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))
	# 		print("This worked")
	# 		print(pytextrank.pretty_print(rl))
def some_func(attr,old,new):
	global value5
	try:
		selected_index = Keyword_text.source.selected.indices[0]
		value5 = str(Keyword_text.source.data["keywords_"][selected_index])
		search_text.value = value5
	except IndexError:
		pass
def my_text_input_handler(attr, old, new):
		print("Previous label: " + old)
		print("Updated label: " + new)
		process_text()
		update_table()
def callback1(event):
	print('Python:Click')
	global value6, value2, value7, table_cell_column_2, dict4, summary3
	dict4 =[]
	
	try:
		selected_index1 = patch.data_source.selected.indices[0]
		value2 = value2 + 1
		value6.append (str(value2)+'. '+str(patch.data_source.data["desc"][selected_index1]) )
		value3 = '\n'.join(value6)
		table_cell_column_2.value = str(value3)

		dict4.append(patch.data_source.data["texts3"][selected_index1])
		print(len(dict4))

		
		clean = sentences1(dict4,1).replace("\\n", " ")

		# Updating summary data table
		value7 = 'Title:' + ' '+ str(patch.data_source.data["desc"][selected_index1]) + '\n' + '\n' + clean
		summary3.value = value7
		

		
		custom_text.value = custom_text.value + "["+str(value2) + "]" 
	except IndexError:
		pass

def function_source(attr, old, new):
	global table_row, table_cell_column_1,value1, value2, value3, value8

	
	
	try:
		
		selected_index = chosen_papers.source.selected.indices[0]
		print(selected_index)
		table_row.value = str(selected_index)
		print(table_row.value)

		for x in value1:
			y = x[3:]
			if y == chosen_papers.source.data["titles_"][selected_index]:
				z = value1.index(x)
				z=z+1
				custom_text.value = custom_text.value + "["+ str(z) + "]" 
				print("hey")
				print(z)

		if not any(chosen_papers.source.data["titles_"][selected_index] in x for x in value1):
			value2 = value2 + 1
			value1.append (str(value2)+'. '+str(chosen_papers.source.data["titles_"][selected_index]) )
			value3 = '\n'.join(value1)
			table_cell_column_1.value = str(value3)
			custom_text.value = custom_text.value + "["+ str(value2) + "]" 
		
				
				

		# if any(y := chosen_papers.source.data["titles_"][selected_index] in x for x in value1):
		# 		print(y)
			# if any(chosen_papers.source.data["titles_"][selected_index] in x):
			# 	custom_text.value = custom_text.value + "["+ x[0] + "]" 
		
		value8 = str(chosen_papers.source.data["authors_"][selected_index])
		whitelist = string.ascii_letters + string.digits + ' ' + ','
		new_s = ''.join(c for c in value8 if c in whitelist)
		search_text.value = new_s
		
		
	except IndexError:
		pass


  

source = []
topic_names = []
def plot_bokeh(title):
	global topic_names, authors
	topic_names = grab_topics(t_model, feature_names, 5)

	x_ = []
	y_ = []
	t_ = []
	o_=[]
	topic_ = []

	global source
	source = ColumnDataSource(
			data =dict(
				x=x_,
				y=y_,
				t=t_,
				o=o_,
				desc=titles,
				topics=topic_,
				auth=authors,
				year=years
				))

	#Draw plots
	update_plot()
	

	global year_from, year_to, custom_text, search_text, radio_search, search_items, Keyword_text, custom_text1

	#Year filtering controls
	year_from = Slider(title="Include papers from", value=1981, start=1981, end=2017, step=1)

	year_to = Slider(title="Inlude papers to", value=2017, start=year_from.value-1, end=2017, step=1)
	year_from.on_change('value', update_years)
	year_to.on_change('value', update_years)
	now_change = Button(label="Update", button_type="success")
	now_change.on_click(update_plot)
	

	#Custom text placement controls
	#sizing_mode = 'scale_both' -- need to look for scaling the text box size
	#sizing_mode = 'stretch_both'
	
	custom_text = TextAreaInput(value=" ", title="Enter some text you are working on here:",width=600,height=400)
	text_button = Button(label="Process", button_type="success")
	# for i in custom_text.value:
	# 	if i == '.':
	# 		process_text()

	# 	callback1 = CustomJS( code="""
	# 	    // the event that triggered the callback is cb_obj:
	# // The event type determines the relevant attributes
	# console.log('Tap event occurred at x-position: ' + cb_obj.x)
	# """)
	
	# custom_text.on_change("value", my_text_input_handler)
	# custom_text.js_on_event('tap', callback1)
	# custom_text.on_event(MouseEnter, callback1)
	text_button.on_click(process_text)
	text_button.on_click(update_table)

	custom_text1 = TextAreaInput(value=" ", title="Enter text here if you want to search for individual words or lines:",width=600,height=200)
	text_button1 = Button(label="Process", button_type="success")
	text_button1.on_click(process_text1)
	text_button1.on_click(update_table)

	template200 = """
	<div style="font-size: 15px;">
	<%= value %>
	</div>
	"""
	
	keyword_data = ColumnDataSource(data ={})
	columns =[TableColumn(field ="keywords_", title ="<b> Keyword </b>", width = 700, formatter=HTMLTemplateFormatter(template=template200))]
	# Keyword_text = TextAreaInput(value ="default",title="Keywords", width = 800, height = 100)
	Keyword_text = DataTable(source=keyword_data, columns=columns, width=800, row_height=50, editable=True, fit_columns = True)

	Keyword_text.source.selected.on_change("indices", some_func)
	






	#Search button controls
	search_text = TextInput(value="", title="Search box: (separate terms with ';' - not for DOIs)")
	search_button = Button(label="Search", button_type="success")
	search_button.on_click(process_search)
	radio_search = RadioButtonGroup(labels=["Title", "Full Text", "doi","Author"], active=1)
	print("helllllll")
	#increase size
	#txtIn_rng = widgetbox(children=[custom_text], width=wd)
	#curdoc().add_root(txtIn_rng)


	#Data Table for selected papers
	
	global table_data, chosen_papers,dict2,table_row, table_cell_column_1, value1, value2, table_cell_column_2, summary3, dict4
	template = """<span href="#" data-toggle="tooltip" title="<%= value %>" style="font-size:15px"><%= value %></span>"""
	template_str = '<a href="http://dx.doi.org/<%=dois_%>" target="_blank"><%= value %></a>'



	# output_file("openurl.html")
	table_data = ColumnDataSource(data = {})
	print("***********")
	columns = [TableColumn(field="years_", title="<b>Year</b>",width = 50, formatter=HTMLTemplateFormatter(template=template)), TableColumn(field="titles_", title="<b>Paper Title</b>", width = 200, formatter=HTMLTemplateFormatter(template=template)), TableColumn(field="authors_", title="<b>Authors</b>", width = 100, formatter=HTMLTemplateFormatter(template=template)), TableColumn(field="dois_", title="<b>Link</b>",width = 100, formatter=HTMLTemplateFormatter(template=template_str)),TableColumn(field="summary_", title="<b>Summary</b>", width =600,formatter=HTMLTemplateFormatter(template=template))]
	print("1*")
	chosen_papers = DataTable(source=table_data, columns=columns, width=600, row_height=100, editable=True, fit_columns = True)

	table_row = TextInput(value = '', title = "Row index:")
	value1 =[]
	value2 = int(0)
	value3 =[]
	dict4 =[]
	table_cell_column_1 = TextAreaInput(value = '', title = "Papers from table", height = 100)
	table_cell_column_2 = TextAreaInput(value = '', title = "Papers from graph", height = 100)
	summary3 = TextAreaInput(value = '', title = "Summary of the recently selected paper", height = 300)
	# table_cell_column_2 = TextInput(value= '', title = "Author", height = 100)

	

	
	chosen_papers.source.selected.on_change('indices', function_source)


	# Adding a title for data table
	pre = Div(text="""Please hover over each column to see the full text. </br>
By clicking on each row, the desirable papers can be added to the 'Papers from table' list below.</br>
By clicking on authors column, authors can be directly placed in search.""",
width=600, height=50, style={'font-size': '100%', 'color': 'black', 'font-weight':'bold'})

# 	pre = PreText(text="""Please hover over each column to see the full text.\n
# By clicking on each row, the desirable papers can be added to the 'Papers from table' list below.\n
# By clicking on authors column, authors can be directly placed in search""",
# 	width=500, height=100)

	# Adding title for keyword data table
	pre1 = Div(text=""" If you click on keywords, the keyword will be directly placed in the 'Search Box' """, width=500, height=20,style={'font-size': '100%', 'color': 'black', 'font-weight':'bold'})







	# chosen_papers.append(HoverTool(tooltips=[("Summary", "@summary_")]))

	# para5 = Paragraph(text = 'summary' , height = 1)

	
	#chosen_papers.source.data.update(update_table())
	# chosen_papers.source.on_change('data', update_table1)


	# global table_data1, chosen_papers1
	# template = """<span href="#" data-toggle="tooltip" title="<%= value %>"><%= value %></span>"""

	# table_data1 = ColumnDataSource(data = {})
	# print("***********")
	# columns = [TableColumn(field="summary_", title="Summary", formatter=HTMLTemplateFormatter(template=template))]
	# print("1*")
	# chosen_papers1 = DataTable(source=table_data, columns=columns, width=800, row_height=100, editable=True,fit_columns = True, scroll_to_selection = True)

	
	# df = pd.DataFrame([
	# 	['this is a longer text that needs a tooltip, because otherwise we do not see the whole text', 'this is a short text'],
	# 	['this is another loooooooooooooooong text that needs a tooltip', 'not much here'],
	# ], columns=['a', 'b'])

	# columns = [TableColumn(field=c, title=c, width=20, formatter=HTMLTemplateFormatter(template=template)) for c in ['a', 'b']]

	# table = DataTable(source=ColumnDataSource(df), columns=columns)

	




	# create a new plot with a title and axis labels
	global patch, value6, value7
	p = figure(title="CHI scatter", tools="tap", x_axis_label='x', y_axis_label='y', width=800, plot_height=850)
	p.add_tools(BoxZoomTool())
	p.add_tools(ZoomInTool())
	p.add_tools(ZoomOutTool())
	p.add_tools(ResetTool())
	p.add_tools(HoverTool(tooltips=[("Title", "@desc"), ("Topic", "@topics"), ("Authors", "@auth"), ("Year", "@year")]))

	patch = p.circle(x='x', y='y', fill_color='t', nonselection_fill_color='t', radius=0.01, fill_alpha='o', nonselection_fill_alpha='o', line_color=None, source=source)
	url = "http://dx.doi.org/@dois"
	taptool = p.select(type=TapTool)
	taptool.callback = OpenURL(url=url)

	value6 = []
	value7 = []
	patch.data_source.on_change('selected', callback)

	p.on_event(Tap, callback1)



	l = gridplot([[ row([column(row([widgetbox(year_from, year_to, now_change), widgetbox(search_text, radio_search, search_button)]),widgetbox(custom_text, text_button), widgetbox(custom_text1, text_button1),widgetbox(pre,chosen_papers))]) ,column([p,widgetbox(pre1,Keyword_text)])  ],[table_cell_column_1], [table_cell_column_2], [summary3]   ])
	curdoc().add_root(l)


	# curdoc().add_root(column([row([widgetbox(year_from, year_to, now_change), widgetbox(custom_text, text_button), widgetbox(search_text, radio_search, search_button)]),p, widgetbox(chosen_papers)]))

#Load papers from database
# load_papers(1981, 2017)

#Load all paper text and details from json files
open_papers(1981, 2017)

#Build new LDA Models
# build_models(1)
# save_models()

#Bypass building models and just load from file
load_models(1)

#Set paper view range to entire corpus
id_interests[1] = len(texts)

#Centre a random paper (the last in the corpus)
new_id = len(texts)-1

#Plot the view
plot_bokeh("veritaps_nmf")
#server.io_loop.start()


