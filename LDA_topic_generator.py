import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
snow = nltk.stem.SnowballStemmer('english')

import re
import sys
import operator
import gensim
from datetime import datetime
from gensim import corpora, models
from collections import defaultdict

# Project authentication stuff to enable access to the data
from google.cloud import bigquery
from google.oauth2 import service_account

# Function for retrieving the current time, used for tracking how long it takes for an operation to finish
def get_time():
   now = datetime.now()
   current_time = now.strftime("%H:%M:%S")
   print("Current Time: ", current_time)

# Function for pre-processing question tags
def preprocess_tags(results_list):
   results_tags = [row[1].split("|") for row in results_list] # Separate the tags from each other
   stemmed_tags = [[snow.stem(word) for word in text] for text in results_tags] # Stem the tags before returning the tagset
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
   stop_words = [word for (word, frequency) in sorted_frequencies[0:quantity_threshold]]

   print("\nRemoving stop words and other formatting artifacts...")
   question_rows_nostop = [[word for word in text if not word in stop_words] for text in question_rows_stemmed]

   # With stop words removed, the question bodies have been pre-processed and are ready to be returned
   return(question_rows_nostop)

# Function for generating LDA topics
def generate_topics(preprocessed, mode):
   if mode == 0:
      flag_lower = "python"
   elif mode == 1:
      flag_lower = "java"
   elif mode == 2:
      flag_lower = "cpp"
   elif mode == 3:
      flag_lower = "js"
   else:
      sys.exit("INVALID OPTION")
   
   # Create dictionary from processed documents containing how many times a word appears in the training set
   preprocessed_dictionary = gensim.corpora.Dictionary(preprocessed)

   # Filter out extreme tokens
   preprocessed_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
   get_time()

   # For each document, create a dictionary reporting how many words it contains and how many times those words appear
   print("\nCreating bow corpora for each document.")
   bow_corpora = [preprocessed_dictionary.doc2bow(text) for text in preprocessed]
   get_time()

   # Running LDA using bag of words
   # Train LDA model
   print("\nTraining LDA model...")
   lda_model = gensim.models.LdaModel(bow_corpora, num_topics=50, id2word=preprocessed_dictionary, passes=2)
   get_time()
   expanded_topics = lda_model.print_topics(-1, 20)
   all_topics = [word_tokenize(re.sub(r'(<.*?>)|[^\w\s]|[0-9_]', ' ', words)) for (index, words) in expanded_topics]
   all_scores = [word_tokenize(re.sub(r'[A-Za-z*"+]', ' ', words)) for (index, words) in expanded_topics]
   all_topics_scores = []

   print(all_topics[0])
   print(all_scores[0])

   words_filename = flag_lower + "_topic_words.txt"
   scores_filename = flag_lower + "_topic_scores.txt"

   f1 = open(words_filename, "w", encoding="utf-8")
   for term_set in all_topics:
      topic_line = " ".join(term_set)
      f1.write(topic_line + "\n")
   f1.close()

   f2 = open(scores_filename, "w", encoding="utf-8")
   for score_set in all_scores:
      score_line = " ".join(score_set)
      f2.write(score_line + "\n")
   f2.close()

   for i in range(len(all_topics)):
      topic_score = list(zip(all_topics[i], [float(score) for score in all_scores[i]]))
      all_topics_scores.append(topic_score)

   for idx, topic in expanded_topics:
      print('Topic: {} \nWords: {}'.format(idx, topic))

   # Return the list of topics and their scores
   return all_topics_scores

def language_operations(results, mode):
   if mode == 0:
      flag_upper = "Python"
   elif mode == 1:
      flag_upper = "Java"
   elif mode == 2:
      flag_upper = "C++"
   elif mode == 3:
      flag_upper = "JavaScript"
   else:
      sys.exit("INVALID OPTION")

   results_list = list(results) # Convert the results into a list
   results_docs = [row[0] for row in results_list] # Isolate the documents themselves from the list
   print("Pre-processing " + flag_upper + " question bodies...")
   preprocessed = preprocess_bodies(results_docs) # Pre-process the question bodies
   print("Pre-processing " + flag_upper + " question tags...")
   stemmed_tags = preprocess_tags(results_list) # Pre-process the questions tags
   get_time()

   print("\n" + flag_upper + " sample results:")
   for result in preprocessed[0:10]:
      print(result)
   get_time()

   print("\nGenerating topics for " + flag_upper + " questions...")
   topics_scores = generate_topics(preprocessed, mode)
   print(topics_scores[0:5])
   get_time()

# Main method
def main():
   project_credentials = service_account.Credentials.from_service_account_file('StackOverflow Dataset Project-046ff1faadbb.json')
   client = bigquery.Client(credentials=project_credentials, project='inner-rhythm-295400')

   # Query for retrieving all the question bodies from the posts_questions table of the StackOverflow dataset that contain "python" in their tags lists, as well as the tags themselves
   python_bodies_query_job = client.query("""
      SELECT body, tags
      FROM StackOverflow.posts_questions
      WHERE REGEXP_EXTRACT(tags, r"python") IS NOT NULL
      LIMIT 100000 """)

   # The same query, but for question bodies containing "java" in their tags lists
   java_bodies_query_job = client.query("""
      SELECT body, tags
      FROM StackOverflow.posts_questions
      WHERE REGEXP_EXTRACT(tags, r"java") IS NOT NULL AND REGEXP_EXTRACT(tags, r"javascript") IS NULL
      LIMIT 100000 """)

   # The same query, but for question bodies containing "c++" in their tags lists
   cpp_bodies_query_job = client.query("""
      SELECT body, tags
      FROM StackOverflow.posts_questions
      WHERE REGEXP_EXTRACT(tags, r"c\+\+") IS NOT NULL
      LIMIT 100000 """)

   # The same query, but for question bodies containing "javascript" or "js" in their tags lists
   js_bodies_query_job = client.query("""
      SELECT body, tags
      FROM StackOverflow.posts_questions
      WHERE REGEXP_EXTRACT(tags, r"javascript") IS NOT NULL OR REGEXP_EXTRACT(tags, r"js") IS NOT NULL
      LIMIT 100000 """)

   python_results = python_bodies_query_job.result() # Retrieve results from the Python bodies query
   java_results = java_bodies_query_job.result() # Retrieve results from the Java bodies query
   cpp_results = cpp_bodies_query_job.result() # Retrieve results from the C++ bodies query
   js_results = js_bodies_query_job.result() # Retrieve results from the JavaScript bodies query

   language_operations(python_results, 0)
   language_operations(java_results, 1)
   language_operations(cpp_results, 2)
   language_operations(js_results, 3)

# Run the program
main()