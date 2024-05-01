#-------------------------------------------------------------------------------
# Name:        PyMLHyperpartisan.py
# Purpose:     detect Hyperpartisan News
#
# Created:     01/28/2024
# Log:         02/14/2024   Changed the .pos_ function to .tag_ as per review comments
#              02/17/2024   Changed the train() method
#              02/29/2024   Added Opendill() and savedill()
#              03/01/2024   Added RandomForest, DecisionTree and KNN classifiers
#              03/11/2024   Added 2 new features - Unique adjective and Unique adverb
# 
#-----
import spacy
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import dill
import os
import warnings
warnings.filterwarnings("ignore")

def opendill(filename='pickles/hyperpartisan_detector.dill'):
    try:
        detector_dump = open(filename, 'rb')
        return detector_dump
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return None

class hyperpartisanDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.models = {
            'RandomForest': RandomForestClassifier(),
            'DecisionTree': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier()
        }
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def preprocess(self, text):
        # Removal of extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Removal of links
        url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                               '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = re.sub(url_regex, '', text)

        # Remove leading and trailing whitespace
        text = text.strip()

        # Remove HTML tags, URLs, and other non-text elements
        text = re.sub(r"<.*?>|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)

        # Convert to lowercase
        text = text.lower()

        return text

    def word_count(self, doc):
        return len(doc)

    def unique_words_count(self, doc):
        return len(set(token.text for token in doc))

    def avg_word_length(self, doc):
        return sum(len(token.text) for token in doc) / len(doc) if len(doc) > 0 else 0

    def sentence_count(self, doc):
        return len(list(doc.sents))

    def verbs_count(self, doc):
        return sum(1 for token in doc if token.tag_ == "VB" or token.tag_ == "VBD" or token.tag_ == "VBG" or token.tag_ == "VBN" or token.tag_ == "VBP" or token.tag_ == "VBZ")

    def adjectives_count(self, doc):
        return sum(1 for token in doc if token.tag_ == "JJ" or token.tag_ == "JJR" or token.tag_ == "JJS")

    def adverbs_count(self, doc):
        return sum(1 for token in doc if token.tag_ == "RB" or token.tag_ == "RBR" or token.tag_ == "RBS")

    def pronouns_count(self, doc):
        return sum(1 for token in doc if token.tag_ == "PRP" or token.tag_ == "PRP$" or token.tag_ == "WP" or token.tag_ == "WP$")

    def conjunctions_count(self, doc):
        return sum(1 for token in doc if token.tag_ == "CC")

    def nouns_count(self, doc):
        return sum(1 for token in doc if token.tag_ == "NN" or token.tag_ == "NNS" or token.tag_ == "NNP" or token.tag_ == "NNPS")

    def entities_count(self, doc):
        return len(doc.ents)

    def stop_words_count(self, doc):
        return sum(1 for token in doc if token.is_stop)

    def punctuation_count(self, doc):
        return sum(1 for token in doc if token.is_punct)

    def digits_count(self, doc):
        return sum(1 for token in doc if token.is_digit)

    def avg_sentence_length(self, doc):
        return len(doc) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0

    def noun_phrases_count(self, doc):
        return len(list(doc.noun_chunks))

    def avg_noun_phrase_length(self, doc):
        return sum(len(chunk.text) for chunk in list(doc.noun_chunks)) / len(list(doc.noun_chunks)) if len(list(doc.noun_chunks)) > 0 else 0

    def avg_entity_length(self, doc):
        return sum(len(ent.text) for ent in list(doc.ents)) / len(list(doc.ents)) if len(list(doc.ents)) > 0 else 0

    def numeric_count(self, doc):
        return sum(1 for token in doc if token.is_digit)

    def quotes_count(self, text):
        return text.count('"') + text.count("'")
    
    def trigram_length(self, doc):
        trigrams = list(zip(doc, doc[1:], doc[2:]))
        return len(trigrams)

    def bigram_length(self, doc):
        bigrams = list(zip(doc, doc[1:]))
        return len(bigrams)
    
    def question_mark_count(self, text):
        return text.count('?')

    def exclamation_count(self, text):
        return text.count('!')
    
    def unique_adjectives(self, doc):
        unique_adj = set()
        for token in doc:
            if token.tag_ in ["JJ", "JJR", "JJS"]:
                unique_adj.add(token.text)
        return len(unique_adj)

    def unique_adverbs(self, doc):
        unique_adv = set()
        for token in doc:
            if token.tag_ in ["RB", "RBR", "RBS"]:
                unique_adv.add(token.text)
        return len(unique_adv)

    def preprocess_and_extract_features(self, df):
        features = []

        for _, row in df.iterrows():
            text = self.preprocess(row['ArticleText'])
            
            # Perform spaCy processing
            doc = self.nlp(text)
            
            # Extract spaCy features
            features.append({
                'Label': row.get('Bias', 'Unknown'),
                'Article_ID': row.get('ArticleID', ''), 
                'word_count': self.word_count(doc),
                'unique_words_count': self.unique_words_count(doc),
                'avg_word_length': self.avg_word_length(doc),
                'sentence_count': self.sentence_count(doc),
                'verbs_count': self.verbs_count(doc),
                'adjectives_count': self.adjectives_count(doc),
                'adverbs_count': self.adverbs_count(doc),
                'pronouns_count': self.pronouns_count(doc),
                'conjunctions_count': self.conjunctions_count(doc),
                'nouns_count': self.nouns_count(doc),
                'entities_count': self.entities_count(doc),
                'stop_words_count': self.stop_words_count(doc),
                'punctuation_count': self.punctuation_count(doc),
                'digits_count': self.digits_count(doc),
                'avg_sentence_length': self.avg_sentence_length(doc),
                'noun_phrases_count': self.noun_phrases_count(doc),
                'avg_noun_phrase_length': self.avg_noun_phrase_length(doc),
                'avg_entity_length': self.avg_entity_length(doc),
                'numeric_count': self.numeric_count(doc),
                'quotes_count': self.quotes_count(text),
                'trigram_length': self.trigram_length(doc),
                'bigram_length': self.bigram_length(doc),
                'question_mark_count': self.question_mark_count(text),
                'exclamation_count': self.exclamation_count(text),
                'unique_adjectives': self.unique_adjectives(doc),
                'unique_adverbs': self.unique_adverbs(doc),
            })

        return pd.DataFrame(features)

    def train(self):
        # Load the dataset - Change the file path if needed.                
        spacy_features_df = pd.read_csv('Datasets\hyperpartisan\Hyperpartisan_Dataset.csv')
        y = spacy_features_df['Label']
        X = spacy_features_df.drop(columns=['Article_ID','Label'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            print(f'{name} trained.')

    def test(self):
        if self.X_test is not None and self.y_test is not None:
            for name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                print(f'{name} Accuracy: {accuracy:.2f}')
                print(f'{name} Classification Report:\n{classification_report(self.y_test, y_pred)}\n')
        else:
            print("No test data available.")

    def predict(self, text):
        text_df = pd.DataFrame({'ArticleText': [text]})
        spacy_features_text = self.preprocess_and_extract_features(text_df)
        spacy_features_text.drop(['Label','Article_ID'], axis=1, inplace=True)

        predictions = {}
        for name, model in self.models.items():
            if name == 'RandomForest':
                prediction = model.predict(spacy_features_text)
                feature_scores = model.predict_proba(spacy_features_text) if hasattr(model, "predict_proba") else None
                feature_scores_str = ','.join(map(str, feature_scores[0])) if feature_scores is not None else "N/A"
                predictions[name] = f'{prediction[0]},{feature_scores_str}'

        return predictions

    def savedill(self, filename='pickles/hyperpartisan_detector.dill'):
        pickle_dir = os.path.dirname(filename)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        
        with open(filename, 'wb') as f:
            dill.dump(self, f)