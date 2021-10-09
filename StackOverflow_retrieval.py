import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
snow = nltk.stem.SnowballStemmer('english')

import os
import re
import sys
import operator
import gensim
import math
from datetime import datetime
from gensim import corpora, models
from collections import defaultdict
from copy import deepcopy

# Project authentication stuff to enable access to the data
from google.cloud import bigquery
from google.oauth2 import service_account

credential_path = "C:\\Users\\vonit\\Downloads\\StackOverflow_analysis\\StackOverflow Dataset Project-046ff1faadbb.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

project_credentials = service_account.Credentials.from_service_account_file('StackOverflow Dataset Project-046ff1faadbb.json')
client = bigquery.Client(credentials=project_credentials, project='inner-rhythm-295400')

nltk_stop_words = list(set(stopwords.words('english')))

# Function for retrieving the current time, used for tracking how long it takes for an operation to finish
def get_time():
   now = datetime.now()
   current_time = now.strftime("%H:%M:%S")
   print("Current Time: ", current_time)

# Function for pre-processing question tags
def preprocess_tags(question_tags):
   split_tags = [row.split("|") for row in question_tags] # Separate the tags from each other
   stemmed_tags = [[snow.stem(word) for word in text] for text in split_tags] # Stem the tags before returning the tagset
   return stemmed_tags

# Function for pre-processing question bodies
def preprocess_bodies(question_bodies):
   # Convert all words to lowercase
   print("\nLowercasing words...")
   question_rows_lower = [text.lower() for text in question_bodies]

   # Remove all words within code tags
   print("\nRemoving program code...")
   question_rows_nocode = [re.sub(r'(?s)(<code>.*?<\/code>)', ' ', text) for text in question_rows_lower]

   # Remove punctuation and formatting tags
   print("\nRemoving punctuation and formatting tags...")
   question_rows_nopunc = [re.sub(r'(<.*?>)|[^\w\s]|[0-9_]', ' ', text) for text in question_rows_nocode]

   # Tokenize words
   print("\nTokenizing words...")
   question_rows_tokens = [word_tokenize(text) for text in question_rows_nopunc]

   # Stem words
   print("\nStemming words...")
   question_rows_stemmed = [[snow.stem(word) for word in text] for text in question_rows_tokens]

   # Remove stop words and other formatting artifacts
   # Combine words from every document into a single list
   all_words = []
   for entry in question_rows_stemmed:
      all_words += entry

   # Create dictionary of individual words and their frequencies
   word_frequencies = defaultdict(lambda : 0)
   for word in all_words:
      word_frequencies[word] += 1

   # Sort the dictionary in descending order of most frequent words
   sorted_frequencies = sorted(word_frequencies.items(), key=operator.itemgetter(1), reverse=True)

   # The top 2% most frequent words will be treated as stop words to be removed
   quantity_threshold = round(len(sorted_frequencies) * 0.02)
   stop_words = list(set([word for (word, frequency) in sorted_frequencies[0:quantity_threshold]] + nltk_stop_words))

   print("\nRemoving stop words and other formatting artifacts...")
   question_rows_nostop = [[word for word in text if not word in stop_words] for text in question_rows_stemmed]

   # With stop words removed, the question bodies have been pre-processed and are ready to be returned
   return(question_rows_nostop)

# Function for reading lines from a topic words or scores file
def read_file(filename):
   f = open(filename, "r") # Open the file
   file_lines = [item[0:-1] for item in f.readlines()] # Read each line from the file, and remove the newline character at the end
   f.close() # Close the file
   lines_separated = [line.split(" ") for line in file_lines] # Break apart the lines at spaces and return them
   return lines_separated

# Function for merging together the words in each topic and their scores
def zip_words_scores(words, scores):
   topics = []
   for i in range(len(words)):
      topic = list(zip(words[i], [float(score) for score in scores[i]]))
      topics.append(topic)
   return topics

# Function for matching topics to individual tags, so that topics can be assigned to tagsets later
def match_topics_tags(tagsets, topics_scores):
   all_tags = []
   print("Compiling list of all distinct tags...")
   for tagset in tagsets:
      all_tags += tagset
   all_distinct_tags = set(all_tags)
   print(str(len(all_distinct_tags)))

   tags_results = {}
   print("Searching topics for tags...")
   for tag in all_distinct_tags:
      # All topics appear to contain completely unique words, so we only need to attach one topic to every tag
      # For each real tag, check if that tag has appeared in any of the topics
      for topic in topics_scores:
         found = False
         # Look in the (word, frequency) pairs of each topic for the tag
         for (word, frequency) in topic:
            if tag == word and found == False:
               # If the tag is present in the tuple, it has been found in that tuple's topic and is highly unlikely to appear in
               # any others
               # The words in each topic are already sorted in descending order of score, so it is sufficient to simply return
               # the first three words of each topic with their scores
               topic_words = [topic_word for (topic_word, probability) in topic]
               tags_results[tag] = topic_words
               break

   return tags_results

# Function for expanding question tagsets
# tagset is a list of tags
# top_tags is a dictionary where the keys are individual tags and the values are lists of tags derived from some topic
def expand_tagset(tagset, top_tags):
   # print("Number of real tags: " + str(len(tagset)))
   
   # Start the expanded tagset with the tags present in the original tagset
   expanded_tagset = []
   expanded_tagset += tagset
   for tag in tagset:
      # Check if the current tag in the initial tagset is a key in the top tags dictionary
      if tag in top_tags:
         extra_tags = top_tags[tag]
         # print("Adding the following extra tags based on " + tag)
         # print(extra_tags)
         expanded_tagset += extra_tags
   # Once all tags in the tagset have been examined, return the expanded tagset
   return expanded_tagset

# Function for getting all distinct words in a collection of documents
def get_distinct_words(preprocessed_docs):
   # Combine words from every document into a single list
   print("Generating inverted index.")
   print("Determining distinct words...")
   all_words = []
   for doc in preprocessed_docs:
      all_words += doc
   all_distinct_words = list(set(all_words))
   return all_distinct_words

# Function for generating queries without additional tags
def generate_standard_queries(title_docs, title_tags):
   title_queries = []
   for title in title_docs:
      base_tagset = title_tags[title_docs.index(title)]
      title_queries.append(list(set(title + base_tagset)))
   return title_queries

# Inactive code
# Alternate function used for automatically deciding what tags to expand queries with
# def generate_expanded_queries(title_docs, title_tags, top_tags):
#    title_queries = []
#    for title in title_docs:
#       base_tagset = title_tags[title_docs.index(title)]
#       title_queries.append(list(set(title + expand_tagset(base_tagset, top_tags))))
#    return title_queries

# Function for getting the document frequencies of each term in a collection of documents
# Part of the process for generating an inverted index for tf-idf weighting
def get_doc_frequencies(preprocessed_docs, all_distinct_words):
   doc_frequencies = defaultdict(lambda : 0)
   for doc in preprocessed_docs:
      doc_set = list(set(doc))
      for word in doc_set:
         doc_frequencies[word] += 1
   return doc_frequencies

# Function for getting the term frequencies of each term in a collection of documents
# Part of the process for generating an inverted index for tf-idf weighting
def get_term_frequencies(preprocessed_docs, all_distinct_words, special_ids):
   term_frequencies = defaultdict(lambda: [])
   for doc in preprocessed_docs:
      doc_set = list(set(doc))
      doc_id = preprocessed_docs.index(doc)
      for word in doc_set:
         term_frequencies[word].append((special_ids[doc_id], doc.count(word)))
   return term_frequencies

# Function for generating tf-idf weights
def generate_tf_idfs(doc, n_docs, doc_freqs, term_freqs, doc_id):
   weighted_doc = []
   individual_tf_idfs = {}

   # Calculate the tf-idf weights of each term in the document
   for word in doc:
      # If the word's tf-idf weight has already been calculated previously, retrieve it and append it with the weight to the
      # weighted document being built
      if word in individual_tf_idfs:
         tf_idf = individual_tf_idfs[word]
         weighted_doc.append((word, tf_idf))
      # If the word's tf-idf weight hasn't already been calculated previously, calculate it manually, add it to the dictionary of
      # found tf-idf weights for each word, and append it with the weight to the weighted document being built
      else:
         # Calculate the idf of the current term in this document
         idf = math.log(n_docs / doc_freqs[word], 2)
         
         for (id, tf) in term_freqs[word]:
            # The document being examined should have an index number equal to doc_id
            if id == doc_id:
               # Calculate the tf-idf weight of this term and append the term with the weight to weighted_doc
               tf_idf = tf * idf
               individual_tf_idfs[word] = tf_idf
               weighted_doc.append((word, tf_idf))
               # Exit this innermost loop and move on to the next term in the document
               break
   return weighted_doc

# Function for generating vector lengths
def generate_vector_length(doc):
   squared_weights = [weight * weight for (word, weight) in doc]
   vector_length_sub = 0
   for weight in squared_weights:
      vector_length_sub += weight
   return math.sqrt(vector_length_sub)

# Function for calculating cosine similarities
def generate_cosine_similarities(queries, answers, tf_idf_answers, vector_lengths):
   cos_sims_list = []
   for query in queries:
      # First, find out which documents share at least one term with the query
      matching_answers = []
      for answer in answers:
         for word in answer:
            if word in query:
               matching_answers.append(answer)
               break

      for answer in matching_answers:
         # To calculate cosine similarity, we need the inner product of the query and the document as well as each of their vector
         # lengths
         # But before we can calculate either of those, we need to assign weights to each term in the query

         # Retrieve the weighted document corresponding to the unweighted matching document being examined and isolate its weights
         original_answer_id = answers.index(answer)
         weighted_answer = tf_idf_answers[original_answer_id]
         target_answer_weights = [weight for (word, weight) in weighted_answer]

         # Add weights to the query from the current document being examined
         weighted_query = []
         retrieved_query_weights = {}
         # Check if each word in the query is also present in the document
         for word in query:
            # If the current word has already received a tf-idf weight, just use that
            if word in retrieved_query_weights:
               weight = retrieved_query_weights[word]
               weighted_query.append((word, weight))
            # If the current word has not received a tf-idf weight, but is present in the document, use the weight from the document
            else:
               if word in answer:
                  weight = target_answer_weights[answer.index(word)]
                  retrieved_query_weights[word] = weight
                  weighted_query.append((word, weight))
               else:
                  # Words only present in the query and not the document have a weight of 0
                  weighted_query.append((word, 0))

         # Calculate the vector_length of the weighted query generated from the query and the document
         query_vector_length = generate_vector_length(weighted_query)

         # To calculate the inner product, we first need to find the intersection of the weighted query and the weighted document
         intersection = [weight for (word, weight) in weighted_query if (word, weight) in weighted_answer]
         inner_product = 0
         for weight in intersection:
            inner_product += weight * weight

         # Finally, to calculate the cosine similarity, divide the inner product by the square root of the product of the vector
         # lengths
         cos_sim = inner_product / (query_vector_length * vector_lengths[original_answer_id])
         cos_sims_list.append((query, answer, cos_sim))
   return cos_sims_list

# Function containing all necessary operations performed on the questions for either one of Python or Java
def language_operations(mode, expansion_limit, base_length):
   # If the mode value is 0, retrieve question bodies from the post_questions table of the StackOverflow dataset that contain "python" in their tags lists as well as the tags themselves
   # If the mode value is 1, do all this with question bodies containing "java" in their tags lists instead
   # If the mode value is 2, do all this with question bodies containing "c++" in their tags lists instead
   # If the mode value is 3, do all this with question bodies containing "javascript" or "js" in their tags lists instead
   if mode == 0:
      flag_upper = "Python"
      flag_lower = "python"
      bodies_query_job = client.query("""
         SELECT body, tags
         FROM StackOverflow.posts_questions
         WHERE REGEXP_EXTRACT(tags, r"python") IS NOT NULL
         LIMIT 100000 """)
   elif mode == 1:
      flag_upper = "Java"
      flag_lower = "java"
      bodies_query_job = client.query("""
         SELECT body, tags
         FROM StackOverflow.posts_questions
         WHERE REGEXP_EXTRACT(tags, r"java") IS NOT NULL AND REGEXP_EXTRACT(tags, r"javascript") IS NULL
         LIMIT 100000 """)
   elif mode == 2:
      flag_upper = "C++"
      flag_lower = "cpp"
      bodies_query_job = client.query("""
         SELECT body, tags
         FROM StackOverflow.posts_questions
         WHERE REGEXP_EXTRACT(tags, r"c\+\+") IS NOT NULL
         LIMIT 100000 """)
   elif mode == 3:
      flag_upper = "JavaScript"
      flag_lower = "js"
      bodies_query_job = client.query("""
         SELECT body, tags
         FROM StackOverflow.posts_questions
         WHERE REGEXP_EXTRACT(tags, r"javascript") IS NOT NULL OR REGEXP_EXTRACT(tags, r"js") IS NOT NULL
         LIMIT 100000 """)
   else:
      sys.exit("INVALID OPTION")
   
   if base_length == 0:
      length_string = "long"
   elif base_length == 1:
      length_string = "short"
   else:
      sys.exit("INVALID OPTION")
      
   results = bodies_query_job.result() # Retrieve results from the bodies query
   results_list = list(results) # Convert the results into a list
   results_docs = [row[0] for row in results_list] # Isolate the documents themselves from the list
   results_tags = [row[1] for row in results_list] # Isolate the tags from the list
   print("Pre-processing " + flag_upper + " question bodies...")
   preprocessed = preprocess_bodies(results_docs) # Pre-process the question bodies
   print("Pre-processing " + flag_upper + " question tags...")
   stemmed_tags = preprocess_tags(results_tags) # Pre-process the question tags
   get_time()

   # The terms and scores in each topic are stored in separate text documents and need to be read separately before being merged together
   print("\nLoading topics for " + flag_upper + " questions...")
   topic_words = read_file(flag_lower + "_topic_words.txt")
   topic_scores = read_file(flag_lower + "_topic_scores.txt")
   topics = zip_words_scores(topic_words, topic_scores)

   print("\nRetrieving top-scoring additional tags for each real " + flag_upper + " tag...")
   top_tags = match_topics_tags(stemmed_tags, topics) # Match topics to individual tags

   # Inactive code used for printing top tags and their associated extra tags
   # print("Tags:")
   # for tag in top_tags:
   #    print(tag)
   # print("Top tags:")
   # for tag in top_tags:
   #    print(top_tags[tag])

   print("Length of " + flag_lower + "_top_tags: " + str(len(top_tags)))
   
   print("\nAssigning topics to " + flag_upper + " tagsets...")

   # Inactive code used to retrieve question titles from the StackOverflow dataset to help determine which ones to use.
   # If the mode value is 0, retrieve question titles from the post_questions table of the StackOverflow dataset that contain "python" in their tags lists as well as the tags themselves
   # If the mode value is 1, do all this with question titles containing "java" in their tags lists instead
   # If the mode value is 2, do all this with question titles containing "c++" in their tags lists instead
   # If the mode value is 3, do all this with question titles containing "javascript" or "js" in their tags lists instead
   # These question titles will be used as queries
   # if mode == 0:
   #    titles_query_job = client.query("""
   #       SELECT title, tags
   #       FROM StackOverflow.posts_questions
   #       WHERE REGEXP_EXTRACT(tags, r"python") IS NOT NULL
   #       LIMIT 80 """)
   # elif mode == 1:
   #    titles_query_job = client.query("""
   #       SELECT title, tags
   #       FROM StackOverflow.posts_questions
   #       WHERE REGEXP_EXTRACT(tags, r"java") IS NOT NULL AND REGEXP_EXTRACT(tags, r"javascript") IS NULL
   #       LIMIT 80 """)
   # elif mode == 2:
   #    titles_query_job = client.query("""
   #       SELECT title, tags
   #       FROM StackOverflow.posts_questions
   #       WHERE REGEXP_EXTRACT(tags, r"c\+\+") IS NOT NULL
   #       LIMIT 80 """)
   # elif mode == 3:
   #    titles_query_job = client.query("""
   #       SELECT title, tags
   #       FROM StackOverflow.posts_questions
   #       WHERE REGEXP_EXTRACT(tags, r"javascript") IS NOT NULL OR REGEXP_EXTRACT(tags, r"js") IS NOT NULL
   #       LIMIT 80 """)
   # else:
   #    sys.exit("INVALID OPTION")
   
   # title_results = titles_query_job.result() # Retrieve results from the titles query
   # title_results_list = list(title_results) # Convert the results into a list
   # title_results_docs = [row[0] for row in title_results_list] # Isolate the titles themselves from the list
   # title_results_tags = [row[1] for row in title_results_list] # Isolate the tags from the list
   # print("Pre-processing " + flag_upper + " question titles...")
   # preprocessed_titles = preprocess_bodies(title_results_docs) # Pre-process the question titles
   # print("Pre-processing " + flag_upper + " question tags...")
   # preprocessed_title_tags = preprocess_tags(title_results_tags) # Pre-process the question tags
   # get_time()

   # for title in title_results_docs:
   #    idx = title_results_docs.index(title)
   #    print("Original title: " + str(title_results_docs[idx]))
   #    print("Original tagset: " + str(title_results_tags[idx]))
   #    print("Preprocessed title: " + str(preprocessed_titles[idx]))
   #    print("Preprocessed tagset: " + str(preprocessed_title_tags[idx]))
   # sys.exit("OK")

   # Load question titles and their real tags
   titles = read_file(flag_lower + "_" + length_string + "_queries.txt")
   titles_joined = [" ".join(title) for title in titles]
   titles_preprocessed = preprocess_bodies(titles_joined)
   title_tags = read_file(flag_lower + "_" + length_string + "_query_tags.txt")
   title_tags_joined = [" ".join(title) for title in title_tags]
   title_tags_preprocessed = preprocess_tags(title_tags_joined)

   # Inactive, outdated code used for displaying read question titles and their tags before and after preprocessing
   # title_docs = long_titles + short_titles
   # title_docs_preprocessed = long_titles_preprocessed + short_titles_preprocessed
   # title_tags = long_title_tags + short_title_tags
   # title_tags_preprocessed = long_title_tags_preprocessed + short_title_tags_preprocessed
   
   # for title in title_docs:
   #    id = title_docs.index(title)
   #    print(title)
   #    print(title_docs_preprocessed[id])
   #    print(title_tags[id])
   #    print(title_tags_preprocessed[id])

   title_queries = generate_standard_queries(titles_preprocessed, title_tags_preprocessed)
   title_queries_exp = deepcopy(title_queries)
   topic_tags = []
   # Fill the topic tags array based on which language the experiment is being carried out on
   # base_length is 0 for long topics and 1 for short topics
   if base_length == 0:
      if mode == 0:
         topic_tags.append(['team', 'width', 'wasn', 'releas', 'height', 'concern', 'storag', 'rectangl', 'trivial', 'corner', 'spent', 'forum', 'platform', 'older', 'processor', 'orient', 'moreov', 'polygon', 'estim', 'websocket'])
         topic_tags.append(['pd', 'queryset', 'phone', 'ugli', 'ago', 'preserv', 'stat', 'flow', 'dialog', 'statist', 'repetit', 'flat', 'seek', 'timezon', 'usb', 'utc', 'hardwar', 'neural', 'vscode', 'autocomplet'])
         topic_tags.append(['center', 'overflow', 'cover', 'poll', 'rememb', 'constraint', 'youtub', 'doubt', 'thrown', 'honest', 'watch', 'tmp', 'scala', 'useless', 'fault', 'race', 'ctype', 'central', 'beforehand', 'overwritten'])
         topic_tags.append(['groupbi', 'twice', 'roll', 'neither', 'instanti', 'busi', 'nor', 'fall', 'annoy', 'postgresql', 'nginx', 'sdk', 'role', 'disappear', 'awesom', 'bundl', 'gunicorn', 'dashboard', 'fork', 'cherrypi'])
         topic_tags.append(['multipli', 'aren', 'overrid', 'shouldn', 'abc', 'creation', 'cross', 'super', 'partial', 'val', 'newlin', 'exceed', 'puzzl', 'necessarili', 'pil', 'novic', 'pls', 'aaa', 'alot', 'epoch'])
         topic_tags.append(['known', 'upgrad', 'difficulti', 'difficult', 'degre', 'disk', 'slash', 'pool', 'purchas', 'boost', 'errno', 'facebook', 'vehicl', 'famili', 'effort', 'jinja', 'architectur', 'openpyxl', 'debian', 'apt'])
         topic_tags.append(['exercis', 'colour', 'near', 'everytim', 'plt', 'walk', 'major', 'correl', 'signific', 'transpar', 'convent', 'discuss', 'fli', 'metadata', 'gpu', 'meta', 'daemon', 'sync', 'subdirectori', 'ping'])
         topic_tags.append(['comparison', 'mail', 'upper', 'pyplot', 'broken', 'act', 'summari', 'co', 'mp', 'financi', 'boundari', 'vm', 'zone', 'std', 'chines', 'didnt', 'tackl', 'ordin', 'runner', 'uniform'])
         topic_tags.append(['ex', 'programm', 'egg', 'extrem', 'noob', 'employe', 'demo', 'scikit', 'feedback', 'ive', 'md', 'checkbox', 'latitud', 'beyond', 'longitud', 'street', 'touch', 'pyinstal', 'accur', 'tail'])
         topic_tags.append(['greater', 'subject', 'keyerror', 'typic', 'substitut', 'icon', 'tcp', 'trial', 'twist', 'dollar', 'suitabl', 'everywher', 'yesterday', 'xxx', 'colab', 'undefin', 'ftp', 'radio', 'viewer', 'paid'])
      elif mode == 1:
         topic_tags.append(['swipe', 'extrem', 'discuss', 'convent', 'mismatch', 'aris', 'weather', 'gestur', 'crazi', 'picker', 'outgo', 'pagin', 'jstl', 'dsl', 'af', 'guidelin', 'finder', 'thankyou', 'gradlew', 'homecontrol'])
         topic_tags.append(['happi', 'summari', 'professor', 'treat', 'jpeg', 'communiti', 'technic', 'afterward', 'ton', 'jwt', 'themselv', 'rabbitmq', 'eclipselink', 'jlist', 'chooser', 'novic', 'bonus', 'nowher', 'micro', 'greet'])
         topic_tags.append(['num', 'triangl', 'ascii', 'divis', 'inde', 'analyz', 'fraction', 'minor', 'applicationcontext', 'forget', 'expens', 'lombok', 'decis', 'trim', 'aka', 'arduino', 'grpc', 'broadcastreceiv', 'eap', 'dedic'])
         topic_tags.append(['teacher', 'omit', 'art', 'jcombobox', 'coverag', 'album', 'clariti', 'mqtt', 'junk', 'restcontrol', 'onchang', 'semest', 'unnam', 'jsonb', 'actionlisten', 'outbound', 'debian', 'sid', 'enlighten', 'dash'])
         topic_tags.append(['broadcast', 'apolog', 'meant', 'preview', 'kept', 'meet', 'androidx', 'quiz', 'hidden', 'loss', 'javas', 'yield', 'techniqu', 'owner', 'paper', 'temperatur', 'invis', 'ff', 'pro', 'famili'])
         topic_tags.append(['broadcast', 'apolog', 'meant', 'preview', 'kept', 'meet', 'androidx', 'quiz', 'hidden', 'loss', 'javas', 'yield', 'techniqu', 'owner', 'paper', 'temperatur', 'invis', 'ff', 'pro', 'famili'])
         topic_tags.append(['dimension', 'margin', 'stub', 'versa', 'breakpoint', 'ca', 'vice', 'typic', 'degre', 'msg', 'trust', 'jnlp', 'fault', 'artist', 'biggest', 'shortest', 'surnam', 'indirect', 'envelop', 'risk'])
         topic_tags.append(['understood', 'rememb', 'textbox', 'immut', 'percentag', 'pin', 'stori', 'registri', 'variant', 'credit', 'bullet', 'cube', 'investig', 'lwjgl', 'plane', 'loos', 'shed', 'reveal', 'onstop', 'rectifi'])
         topic_tags.append(['dimens', 'univers', 'pure', 'inflat', 'inspect', 'analysi', 'enforc', 'volley', 'backward', 'opt', 'intersect', 'seek', 'prototyp', 'checker', 'gray', 'door', 'oom', 'occas', 'fat', 'rare'])
         topic_tags.append(['simplest', 'vendor', 'mutat', 'usr', 'runwork', 'bufferedimag', 'permit', 'candid', 'elast', 'motion', 'driven', 'frameworkservlet', 'coyot', 'checkstyl', 'horribl', 'jrubi', 'internaldofilt', 'ru', 'abstractprotocol', 'unsatisfi'])
      elif mode == 2:
         topic_tags.append(['portion', 'discov', 'challeng', 'dispatch', 'capabl', 'cdecl', 'consecut', 'connector', 'shortest', 'experiment', 'protobuf', 'facet', 'repo', 'du', 'webpag', 'nm', 'bank', 'dlopen', 'simd', 'proto'])
         topic_tags.append(['str', 'plus', 'simpler', 'ultim', 'blob', 'geometri', 'closest', 'enclos', 'qdialog', 'honest', 'standalon', 'plot', 'xxx', 'unchang', 'messagebox', 'knew', 'yourself', 'qgraphicsscen', 'hover', 'threadpool'])
         topic_tags.append(['illustr', 'qstring', 'factor', 'restart', 'expens', 'met', 'reflect', 'deprec', 'reinterpret', 'cgal', 'rand', 'compos', 'trail', 'expert', 'iso', 'radius', 'alias', 'sit', 'forbid', 'rtti'])
         topic_tags.append(['intersect', 'lack', 'vec', 'stage', 'uniform', 'chart', 'paragraph', 'permut', 'nan', 'scheme', 'enlighten', 'realis', 'asid', 'outlin', 'idl', 'libboost', 'defer', 'decltyp', 'expr', 'educ'])
         topic_tags.append(['emit', 'belong', 'chanc', 'sphere', 'among', 'vari', 'paper', 'recompil', 'incompat', 'alon', 'offici', 'occupi', 'batch', 'optimis', 'statist', 'indent', 'contour', 'coeffici', 'clion', 'myfil'])
         topic_tags.append(['bracket', 'sizeof', 'worri', 'concret', 'repli', 'unexpect', 'watch', 'satisfi', 'dot', 'osx', 'prefix', 'kb', 'assist', 'frequenc', 'mt', 'fastest', 'naiv', 'frustrat', 'volatil', 'tend'])
         topic_tags.append(['getter', 'setter', 'pod', 'toolchain', 'crc', 'natur', 'plain', 'lie', 'elsewher', 'bring', 'rc', 'aim', 'ton', 'wstring', 'flash', 'raii', 'reload', 'licens', 'modal', 'chat'])
         topic_tags.append(['scroll', 'cpprefer', 'impli', 'grab', 'highest', 'weight', 'dumb', 'lowest', 'facil', 'rearrang', 'greatest', 'needless', 'allegro', 'msys', 'freetyp', 'speedup', 'bone', 'granular', 'discourag', 'slowdown'])
         topic_tags.append(['discuss', 'besid', 'five', 'comma', 'deleg', 'latter', 'fire', 'deploy', 'abort', 'kept', 'narrow', 'iostream', 'anonym', 'filesystem', 'former', 'balanc', 'profession', 'sh', 'unlik', 'usabl'])
         topic_tags.append(['act', 'smallest', 'dereferenc', 'proxi', 'worth', 'typenam', 'decreas', 'reader', 'messi', 'redund', 'nullptr', 'strategi', 'writer', 'drawback', 'webcam', 'restor', 'resort', 'movi', 'marker', 'deseri'])
      elif mode == 3:
         topic_tags.append(['rid', 'compat', 'lightbox', 'requirej', 'relationship', 'exercis', 'power', 'hex', 'gather', 'measur', 'circular', 'isotop', 'startup', 'boot', 'ca', 'backward', 'funtion', 'onsubmit', 'masonri', 'oracl'])
         topic_tags.append(['onload', 'syntaxerror', 'suspect', 'aim', 'hang', 'branch', 'md', 'alt', 'fullscreen', 'propos', 'clipboard', 'jstl', 'plane', 'pipelin', 'showcas', 'jpeg', 'ecmascript', 'partner', 'birthday', 'commonj'])
         topic_tags.append(['rid', 'compat', 'lightbox', 'requirej', 'relationship', 'exercis', 'power', 'hex', 'gather', 'measur', 'circular', 'isotop', 'startup', 'boot', 'ca', 'backward', 'funtion', 'onsubmit', 'masonri', 'oracl'])
         topic_tags.append(['ive', 'versa', 'vice', 'timezon', 'myapp', 'zone', 'handlebar', 'rerend', 'portal', 'lag', 'teacher', 'bl', 'ock', 'mapbox', 'buggi', 'stackblitz', 'hbs', 'gl', 'ternari', 'ecommerc'])
         topic_tags.append(['variat', 'rememb', 'doubt', 'programmat', 'song', 'latitud', 'longitud', 'averag', 'music', 'satisfi', 'worri', 'toward', 'stripe', 'overlook', 'tcp', 'motion', 'central', 'daili', 'havent', 'factor'])
         topic_tags.append(['onchang', 'injector', 'unknown', 'mix', 'spent', 'seper', 'meet', 'defer', 'shot', 'lazi', 'unpr', 'wich', 'balanc', 'sqlite', 'tsx', 'pro', 'webserv', 'gon', 'rect', 'furthermor'])
         topic_tags.append(['stick', 'appl', 'criteria', 'revert', 'recreat', 'sake', 'throughout', 'poor', 'pin', 'train', 'geoloc', 'webapi', 'lookup', 'accur', 'monday', 'dropzon', 'repetit', 'leak', 'pay', 'excerpt'])
         topic_tags.append(['lose', 'vanilla', 'curious', 'eventu', 'hasn', 'currenc', 'sync', 'linux', 'btn', 'deprec', 'scrape', 'ee', 'spread', 'tostr', 'attack', 'hyphen', 'cb', 'preset', 'patch', 'placement'])
         topic_tags.append(['iostream', 'deleg', 'filesystem', 'deploy', 'abort', 'latter', 'fire', 'discuss', 'besid', 'five', 'comma', 'kept', 'narrow', 'anonym', 'former', 'balanc', 'profession', 'sh', 'unlik', 'usabl'])
         topic_tags.append(['act', 'mine', 'belong', 'initialis', 'modern', 'owner', 'gson', 'jsonobject', 'eye', 'javax', 'fulfil', 'nt', 'favorit', 'exchang', 'uppercas', 'postback', 'alongsid', 'emoji', 'pager', 'commerc'])
      else:
         sys.exit("INVALID OPTION")
   elif base_length == 1:
      if mode == 0:
         topic_tags.append(['delimit', 'pivot', 'sound', 'kill', 'lock', 'dist', 'frequent', 'backslash', 'registr', 'hey', 'ton', 'scienc', 'aris', 'startup', 'iloc', 'portfolio', 'scene', 'wav', 'februari', 'tax'])
         topic_tags.append(['front', 'welcom', 'disabl', 'lose', 'slight', 'everyon', 'freez', 'nativ', 'modif', 'chat', 'sudo', 'exchang', 'varieti', 'cx', 'nicer', 'fresh', 'tricki', 'exponenti', 'quicker', 'sensit'])
         topic_tags.append(['legend', 'indent', 'vari', 'num', 'reset', 'board', 'threshold', 'movi', 'archiv', 'manner', 'afterward', 'lat', 'station', 'genr', 'incorpor', 'tie', 'forth', 'unzip', 'lon', 'decis'])
         topic_tags.append(['delimit', 'pivot', 'sound', 'kill', 'lock', 'dist', 'frequent', 'backslash', 'registr', 'hey', 'ton', 'scienc', 'aris', 'startup', 'iloc', 'portfolio', 'scene', 'wav', 'februari', 'tax'])
         topic_tags.append(['anywher', 'okay', 'especi', 'mechan', 'silli', 'navig', 'complain', 'coin', 'ms', 'subscript', 'truncat', 'wast', 'fixtur', 'discard', 'phase', 'immut', 'argpars', 'udp', 'realis', 'shop'])
         topic_tags.append(['multipli', 'aren', 'overrid', 'shouldn', 'abc', 'creation', 'cross', 'super', 'partial', 'val', 'newlin', 'exceed', 'puzzl', 'necessarili', 'pil', 'novic', 'pls', 'aaa', 'alot', 'epoch'])
         topic_tags.append(['shift', 'onto', 'ensur', 'contact', 'arrow', 'spot', 'multiindex', 'volum', 'rough', 'datatyp', 'carri', 'amazon', 'inclus', 'hasn', 'dt', 'gaussian', 'cento', 'asset', 'semi', 'bs'])
         topic_tags.append(['behind', 'percentag', 'blue', 'scope', 'straight', 'auth', 'easiest', 'met', 'grate', 'marker', 'conveni', 'outer', 'sklearn', 'spars', 'deprec', 'reli', 'span', 'redi', 'spell', 'social'])
         topic_tags.append(['pd', 'queryset', 'phone', 'ugli', 'ago', 'preserv', 'stat', 'flow', 'dialog', 'statist', 'repetit', 'flat', 'seek', 'timezon', 'usb', 'utc', 'hardwar', 'neural', 'vscode', 'autocomplet'])
         topic_tags.append(['math', 'concept', 'jpg', 'inner', 'challeng', 'eventu', 'scan', 'ultim', 'experienc', 'parti', 'ten', 'ab', 'redund', 'hidden', 'parenthesi', 'prior', 'shall', 'inc', 'uwsgi', 'mismatch'])
      elif mode == 1:
         topic_tags.append(['fatal', 'imgur', 'gotten', 'kinda', 'illegalstateexcept', 'obj', 'bellow', 'bat', 'barcod', 'col', 'wire', 'arrayadapt', 'diagon', 'con', 'edu', 'chess', 'pros', 'market', 'backtrack', 'junior'])
         topic_tags.append(['pi', 'gridview', 'serializ', 'dropdown', 'thx', 'cardview', 'trial', 'progressbar', 'ouput', 'reflectivemethodinvoc', 'raspberri', 'drool', 'kubernet', 'mirror', 'cj', 'explod', 'usabl', 'hikari', 'pod', 'medic'])
         topic_tags.append(['illeg', 'chosen', 'ive', 'pastebin', 'sharedprefer', 'settext', 'gif', 'newer', 'ed', 'cheer', 'flash', 'smart', 'teach', 'lat', 'opposit', 'useless', 'knew', 'wallpap', 'filesystem', 'viewhold'])
         topic_tags.append(['multipli', 'instrument', 'pid', 'puzzl', 'broken', 'eventdispatchthread', 'yaml', 'helloworld', 'predefin', 'colleg', 'unlock', 'aaa', 'jna', 'webserv', 'perman', 'accesscontrol', 'matlab', 'protectiondomain', 'actionperform', 'doprivileg'])
         topic_tags.append(['ioexcept', 'wifi', 'surpris', 'ship', 'lab', 'overal', 'acquir', 'gist', 'arrang', 'seat', 'race', 'bypass', 'udp', 'ambigu', 'truli', 'dns', 'deliveri', 'uml', 'monday', 'mono'])
         topic_tags.append(['transpar', 'toolbar', 'discov', 'annoy', 'strong', 'npe', 'silli', 'yesterday', 'spawn', 'angl', 'robot', 'crop', 'perspect', 'hive', 'findviewbyid', 'resteasi', 'surefir', 'kick', 'alright', 'polymorph'])
         topic_tags.append(['swipe', 'extrem', 'discuss', 'convent', 'mismatch', 'aris', 'weather', 'gestur', 'crazi', 'picker', 'outgo', 'pagin', 'jstl', 'dsl', 'af', 'guidelin', 'finder', 'thankyou', 'gradlew', 'homecontrol'])
         topic_tags.append(['guidanc', 'tester', 'grow', 'apart', 'hasn', 'trick', 'formatt', 'fresh', 'docx', 'realm', 'occupi', 'gap', 'wav', 'wide', 'honest', 'gridlayout', 'bother', 'dig', 'unpars', 'fulfil'])
         topic_tags.append(['destroy', 'tap', 'slide', 'var', 'classcastexcept', 'vari', 'align', 'hook', 'sit', 'bash', 'park', 'frustrat', 'diagram', 'grail', 'mod', 'formal', 'outputstream', 'minecraft', 'shoot', 'tail'])
         topic_tags.append(['anonym', 'runtimeexcept', 'deriv', 'hashcod', 'compareto', 'compos', 'poll', 'programat', 'alertdialog', 'glide', 'scheme', 'tick', 'reopen', 'pet', 'intens', 'forth', 'pie', 'livedata', 'tini', 'sudo'])
      elif mode == 2:
         topic_tags.append(['middl', 'physic', 'modern', 'legaci', 'simplest', 'adjac', 'quad', 'overal', 'publish', 'seg', 'volum', 'invert', 'overlook', 'motiv', 'parenthesi', 'stdin', 'visualstudio', 'intens', 'viewer', 'refus'])
         topic_tags.append(['ugli', 'unnecessari', 'pad', 'higher', 'archiv', 'broken', 'wise', 'incom', 'threshold', 'escap', 'emul', 'feed', 'spend', 'reciev', 'stdout', 'race', 'ed', 'verbos', 'distinguish', 'multidimension'])
         topic_tags.append(['analysi', 'demo', 'wall', 'abc', 'border', 'disappear', 'demonstr', 'forgot', 'wm', 'refresh', 'paint', 'notif', 'manifest', 'phrase', 'migrat', 'aliv', 'accumul', 'hwnd', 'unari', 'sync'])
         topic_tags.append(['chain', 'datatyp', 'codeblock', 'branch', 'reus', 'guard', 'carri', 'chosen', 'dequ', 'sensor', 'hear', 'shorter', 'popup', 'decis', 'fresh', 'overcom', 'walk', 'backtrac', 'sigsegv', 'cap'])
         topic_tags.append(['middl', 'physic', 'modern', 'legaci', 'simplest', 'adjac', 'quad', 'overal', 'publish', 'seg', 'volum', 'invert', 'overlook', 'motiv', 'parenthesi', 'stdin', 'visualstudio', 'intens', 'viewer', 'refus'])
         topic_tags.append(['greater', 'bigger', 'arg', 'printf', 'sentenc', 'largest', 'price', 'arrow', 'strict', 'linkag', 'libstdc', 'grow', 'bag', 'versa', 'logger', 'polici', 'mm', 'vice', 'theori', 'palindrom'])
         topic_tags.append(['school', 'center', 'pseudo', 'apolog', 'na', 'period', 'formula', 'corner', 'abil', 'wan', 'spent', 'kinda', 'autom', 'timeout', 'ssl', 'materi', 'wheel', 'investig', 'curv', 'wherev'])
         topic_tags.append(['destruct', 'straight', 'drawn', 'qtcreator', 'upgrad', 'upper', 'propos', 'openssl', 'thousand', 'conclus', 'isol', 'green', 'navig', 'ahead', 'recreat', 'rare', 'caught', 'mismatch', 'exclud', 'convinc'])
         topic_tags.append(['illustr', 'qstring', 'factor', 'restart', 'expens', 'met', 'reflect', 'deprec', 'reinterpret', 'cgal', 'rand', 'compos', 'trail', 'expert', 'iso', 'radius', 'alias', 'sit', 'forbid', 'rtti'])
         topic_tags.append(['destruct', 'straight', 'drawn', 'qtcreator', 'upgrad', 'upper', 'propos', 'openssl', 'thousand', 'conclus', 'isol', 'green', 'navig', 'ahead', 'recreat', 'rare', 'caught', 'mismatch', 'exclud', 'convinc'])
      elif mode == 3:
         topic_tags.append(['school', 'round', 'age', 'mp', 'closest', 'extj', 'clock', 'reject', 'plug', 'clarif', 'builder', 'amazon', 'utc', 'secret', 'xxx', 'curv', 'pain', 'constraint', 'abort', 'expressj'])
         topic_tags.append(['rewrit', 'val', 'referenceerror', 'uri', 'signup', 'movement', 'discuss', 'col', 'boundari', 'multipli', 'req', 'fruit', 'album', 'con', 'artist', 'distribut', 'boilerpl', 'submenus', 'dir', 'bother'])
         topic_tags.append(['okay', 'equival', 'webapp', 'opinion', 'millisecond', 'setter', 'getter', 'expir', 'scan', 'solid', 'tablet', 'npmjs', 'fundament', 'earli', 'violat', 'contract', 'landscap', 'vb', 'chines', 'portrait'])
         topic_tags.append(['obj', 'somewhat', 'subsequ', 'five', 'him', 'expert', 'compos', 'architectur', 'technolog', 'xyz', 'firstnam', 'carri', 'parallax', 'interceptor', 'abstract', 'decor', 'userid', 'trial', 'stumbl', 'upward'])
         topic_tags.append(['stick', 'appl', 'criteria', 'revert', 'recreat', 'sake', 'throughout', 'poor', 'pin', 'train', 'geoloc', 'webapi', 'lookup', 'accur', 'monday', 'dropzon', 'repetit', 'leak', 'pay', 'excerpt'])
         topic_tags.append(['onchang', 'injector', 'unknown', 'mix', 'spent', 'seper', 'meet', 'defer', 'shot', 'lazi', 'unpr', 'wich', 'balanc', 'sqlite', 'tsx', 'pro', 'webserv', 'gon', 'rect', 'furthermor'])
         topic_tags.append(['stick', 'appl', 'criteria', 'revert', 'recreat', 'sake', 'throughout', 'poor', 'pin', 'train', 'geoloc', 'webapi', 'lookup', 'accur', 'monday', 'dropzon', 'repetit', 'leak', 'pay', 'excerpt'])
         topic_tags.append(['demonstr', 'draggabl', 'de', 'lock', 'awesom', 'rais', 'underneath', 'dump', 'ubuntu', 'shall', 'communiti', 'mount', 'major', 'ultim', 'medium', 'flexibl', 'restor', 'deactiv', 'newtonsoft', 'droppabl'])
         topic_tags.append(['odd', 'zip', 'algorithm', 'potenti', 'alon', 'strict', 'recip', 'composit', 'consider', 'phase', 'formdata', 'mutat', 'deselect', 'guest', 'loopback', 'assumpt', 'dinam', 'choosen', 'nut', 'unus'])
         topic_tags.append(['rid', 'compat', 'lightbox', 'requirej', 'relationship', 'exercis', 'power', 'hex', 'gather', 'measur', 'circular', 'isotop', 'startup', 'boot', 'ca', 'backward', 'funtion', 'onsubmit', 'masonri', 'oracl'])
      else:
         sys.exit("INVALID OPTION")

   for i in range(10):
      extra_tags = topic_tags[i] # Retrieve the terms/tags to be added to the query
      title_queries_exp[i] += extra_tags[0:expansion_limit] # Add the specified number of terms to the query
      title_queries_exp[i] = list(set(title_queries_exp[i])) # Remove possible duplicate terms from the query after adding them to it

   # Inactive code
   # Query for retrieving answers from the posts_answers table of the StackOverflow dataset
   # answers_query_job = client.query("""
   #    SELECT id, body
   #    FROM StackOverflow.posts_answers
   #    LIMIT 200000 """)
   # answers_list = list(answers_query_job.result())
   # answer_ids = [row[0] for row in answers_list]
   # answer_bodies = [row[1] for row in answers_list]
   # answers_preprocessed = preprocess_bodies(answer_bodies)
   # for id in answer_ids[0:10]:
   #    print(id)

   
   # Inactive code for writing answers to documents
   # for id in answer_ids:
   #    filename = "answer_docs/" + str(id) + ".txt"
   #    f = open(filename, "w", encoding="utf-8")
   #    f.write(answer_bodies[answer_ids.index(id)])
   #    f.close()
   # sys.exit("OK")

   # Read answer info from documents
   filenames = os.listdir("answer_docs")
   answer_ids = []
   answer_bodies = []
   for file in filenames:
      answer_ids.append(file[0:-4])
      filename = "answer_docs/" + file
      f = open(filename, "r", encoding="utf-8")
      answer_bodies.append(f.read())
      f.close()
   answers_preprocessed = preprocess_bodies(answer_bodies)

   # Prepare to calculate tf-idf weights by building an inverted index
   distinct_answer_words = get_distinct_words(answers_preprocessed)
   answer_dfs = get_doc_frequencies(answers_preprocessed, distinct_answer_words)
   answer_tfs = get_term_frequencies(answers_preprocessed, distinct_answer_words, answer_ids)

   # for word in distinct_answer_words:
   #    print(word + " appears in " + str(answer_dfs[word]) + " documents.")
   #    for (id, count) in answer_tfs[word]:
   #       print(word + " appears in document " + str(id) + " " + str(count) + " times.")

   # With the inverted index, we can assign tf-idf weights and vector lengths to each answer.
   print("There are " + str(len(answers_preprocessed)) + " answer documents.")
   print("Calculating tf-idf weights for each answer...")
   tf_idf_answers = []
   for answer in answers_preprocessed:
      weighted_answer = generate_tf_idfs(answer, len(answers_preprocessed), answer_dfs, answer_tfs, answer_ids[answers_preprocessed.index(answer)])
      tf_idf_answers.append(weighted_answer)

   print("Calculating vector lengths for each document...")
   vector_lengths = []
   for answer in tf_idf_answers:
      vector_length = generate_vector_length(answer)
      vector_lengths.append(vector_length)

   # Calculate cosine similarities between questions and answers
   print("Calculating cosine similarities between queries and answers...")
   cosine_similarities = generate_cosine_similarities(title_queries_exp, answers_preprocessed, tf_idf_answers, vector_lengths)
   print("Cosine similarities for " + length_string + " " + flag_upper + " queries")
   for query in title_queries_exp:
      query_id = title_queries_exp.index(query)
      # Only cosine similarities less than 1.0 will be taken into account for now so that the results aren't skewed
      # This should only affect a handful of outlier query-answer pairs whose cosine similarity values are extremely slightly higher than 1.0
      target_cosine_similarities = [(title_queries_exp.index(target_query) + 1, answer_ids[answers_preprocessed.index(answer)], cos_sim) for (target_query, answer, cos_sim) in cosine_similarities if title_queries_exp.index(target_query) == query_id and cos_sim <= 1.0]
      distinct_cosine_similarities = list(set([i for i in target_cosine_similarities]))
      sorted_cosine_similarities = sorted(distinct_cosine_similarities, key = lambda x: x[2], reverse = True)

      print("Query: ")
      print(query)
      num_answers_found = len(sorted_cosine_similarities)
      print(str(num_answers_found) + " results out of 200000 were found.")
      if num_answers_found > 50:
         printed_results_length = 50
      else:
         printed_results_length = num_answers_found
      if printed_results_length > 0:
         for (target_query, answer, cos_sim) in sorted_cosine_similarities[:printed_results_length]:
            print("Query No.: " + str(target_query) + ", Answer ID: " + str(answer) + ", Cosine Similarity: " + str(cos_sim))
         isolated_cosine_similarities = [cos_sim for (target_query, answer, cos_sim) in sorted_cosine_similarities[:printed_results_length]]
         total_cos_sim = 0
         for i in range(len(isolated_cosine_similarities)):
            total_cos_sim += isolated_cosine_similarities[i]
         
         avg_cos_sim = total_cos_sim / printed_results_length
         if printed_results_length == 50:
            print("Average cosine similarity for the top 50 answers found for this query: " + str(avg_cos_sim))
         else:
            print("Average cosine similarity for all answers found for this query: " + str(avg_cos_sim))

# Main method
def main():
   # Perform the necessary operations on each language
   # The first command line parameter is 0 for Python, 1 for Java, 2 for C++, 3 for JavaScript
   # The second command line parameter is 0, 5, 10, 15, or 20 (the number of tags to add in the experiment)
   # The third command line parameter is 0 (for long queries) or 1 (for short queries)
   get_time()
   language_operations(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
   get_time()

# Run the program
main()
