#!/usr/bin/env python
# coding: utf-8

# #### Download the XML Dataest from the link:  https://zenodo.org/records/1489920
# I have used the below dataset files from the link for the supervised classification
# 1. articles-training-bypublisher-20181122.zip
# 2. ground-truth-training-bypublisher-20181122.zip 
# 
# Refer the two .xsd files for headers - article.xsd and ground-truth.xsd for converting XML to CSV 
# 
# Change all the file path variables with the local storage path of the XML files accordingly.
# I have renamed articles-training-bypublisher-20181122.xml to "Article.xml" and ground-truth-training-bypublisher-20181122.xml to "Groundtruth.xml"

import pandas as pd
import spacy
import re
import xml.etree.ElementTree as ET
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# Define a function to recursively extract text content from XML elements
def extract_text(element):
    text = ''
    for child in element:
        text += extract_text(child)
    if element.text:
        text += element.text
    return text

# Convert articles dataset XML to TSV
xml_training_file_path = 'Article.xml'
tsv_training_file_path = 'Article.tsv'

# Convert ground truth XML to TSV
xml_groundtruth_train_fp = 'Groundtruth.xml'
tsv_groundtruth_train_fp = 'Groundtruth.tsv'


def xml_to_tsv(xml_file_path, tsv_file_path):
    # Open and parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Open the TSV file for writing
    with open(tsv_file_path, 'w', encoding='utf-8') as tsvfile:
        # Write the headers
        tsvfile.write("ArticleID\tTitle\tPublishedAt\tArticleText\n")

        # Iterate through 'article' elements
        for article in root.findall('.//article'):
            article_id = article.get('id')
            title = article.get('title', '')
            published_at = article.get('published-at', '')
            article_text = extract_text(article)

            # Write the data to the TSV file
            tsvfile.write(f"{article_id}\t{title}\t{published_at}\t{article_text}\n")

    print(f"Article XML data has been successfully converted to {tsv_file_path}.")


def xml_to_tsv_groundtruth(xml_file_path, tsv_file_path):
    # Open and parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Open the TSV file for writing
    with open(tsv_file_path, 'w', encoding='utf-8') as tsvfile:
        # Write the headers
        tsvfile.write("ArticleID\tURL\tHyperpartisan\tBias\tLabeledBy\n")

        # Iterate through 'article' elements
        for article in root.findall('.//article'):
            article_id = article.get('id')
            url = article.get('url', '')
            hyperpartisan = article.get('hyperpartisan', '')
            bias = article.get('bias', '')
            labeled_by = article.get('labeled-by', '')

            # Write the data to the TSV file
            tsvfile.write(f"{article_id}\t{url}\t{hyperpartisan}\t{bias}\t{labeled_by}\n")

    print(f"Ground truth XML data has been successfully converted to {tsv_file_path}.")


current_time = datetime.now()
print(f"Start time: {current_time} ")

# Call the function to convert article data to TSV
xml_to_tsv(xml_training_file_path, tsv_training_file_path)
current_time = datetime.now()
print(f"End time: {current_time} ")


# Call the function to convert ground truth data to CSV
xml_to_tsv_groundtruth(xml_groundtruth_train_fp, tsv_groundtruth_train_fp)

current_time = datetime.now()
print(f"End time: {current_time} ")

#Converting TSV to CSV 
tsv_training_file_path = "Article.tsv"
tsv_groundtruth_train_fp = "Groundtruth.tsv"
csv_training_file_path = "Article.csv"
csv_groundtruth_train_fp = "Groundtruth.csv"

df_article_train = pd.read_csv(tsv_training_file_path, sep='\t')
df_ground_truth_train = pd.read_csv(tsv_groundtruth_train_fp, sep='\t')

#Convert Dataframe to CSV and save it local directory for further usage
df_article_train.to_csv(csv_training_file_path, index=False)
df_ground_truth_train.to_csv(csv_groundtruth_train_fp, index=False)

df_article_train = pd.read_csv('Article.csv')
print("Data Loading Completed")

df_ground_truth_train = pd.read_csv('Groundtruth.csv')
print("Data Loading Completed")


#Data Cleaning - Article ID
is_null = df_article_train['ArticleID'].isnull()

# Count the number of null values in the column
null_count = df_article_train['ArticleID'].isnull().sum()

# Display the results
print("Null values in the column:", is_null)
print("Number of null values in the column:", null_count)

# Check for non-numeric values
non_numeric_values = df_article_train[pd.to_numeric(df_article_train['ArticleID'], errors='coerce').notna()]
non_numeric_count = pd.to_numeric(df_article_train['ArticleID'], errors='coerce').notna().sum()

print("\nNon-numeric values in the count:")
print(non_numeric_count)

df_article_train = non_numeric_values.copy()


#Data Cleaning - ArticleText

is_null = df_article_train['ArticleText'].isnull()

# Count the number of null values in the column
null_count = df_article_train['ArticleText'].isnull().sum()

# Display the results
print("Null values in the column:", is_null)
print("Number of null values in the column:", null_count)

df_article_train.dropna(subset=['ArticleText'], inplace=True)

# Count the number of null values in the column
null_count = df_article_train['ArticleText'].isnull().sum()
print("Number of null values in the column:", null_count)

#Data Cleaning - Bias

is_null = df_ground_truth_train['Bias'].isnull()

# Count the number of null values in the column
null_count = df_ground_truth_train['Bias'].isnull().sum()

# Display the results
print("Null values in the column:", is_null)
print("Number of null values in the column:", null_count)


# Get unique values in the 'Bias' column
unique_bias_values = df_ground_truth_train['Bias'].unique()

# Display unique values
print("Unique values in the 'Bias' column:", unique_bias_values)


is_null = df_ground_truth_train['ArticleID'].isnull()

# Count the number of null values in the column
null_count = df_ground_truth_train['ArticleID'].isnull().sum()

# Display the results
print("Null values in the column:", is_null)
print("Number of null values in the column:", null_count)


# Check for non-numeric values
non_numeric_values = df_ground_truth_train[pd.to_numeric(df_ground_truth_train['ArticleID'], errors='coerce').isna()]
non_numeric_count = pd.to_numeric(df_ground_truth_train['ArticleID'], errors='coerce').isna().sum()

print("\nNon-numeric values in the column:")
print(non_numeric_values)
print("\nNon-numeric values in the count:")
print(non_numeric_count)


# Convert articleid column in df1 to int64
df_article_train['ArticleID'] = df_article_train['ArticleID'].astype('int64')
# Convert articleid column in df1 to int64
df_ground_truth_train['ArticleID'] = df_ground_truth_train['ArticleID'].astype('int64')


# Merge based on 'articleid'
merged_df = pd.merge(df_article_train, df_ground_truth_train, on='ArticleID', how='inner')


merged_df.drop(['PublishedAt', 'URL', 'LabeledBy'  ], axis=1, inplace=True)

merged_df.to_csv('Merged_march2.csv', index=False)

merged_df.fillna('', inplace=True)
        
# Mapping of old labels to new labels to increase efficiency
label_mapping = {'left-center': 'left', 'right-center': 'right'}

# Replace labels in the 'Bias' column using the mapping
merged_df['Bias'] = merged_df['Bias'].replace(label_mapping)

# Save the updated DataFrame to a new CSV file
merged_df.to_csv('label_merged_final.csv', index=False)

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
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

def word_count(doc):
    return len(doc)

def unique_words_count(doc):
    return len(set(token.text for token in doc))

def avg_word_length(doc):
    return sum(len(token.text) for token in doc) / len(doc) if len(doc) > 0 else 0

def sentence_count(doc):
    return len(list(doc.sents))

def verbs_count(doc):
    return sum(1 for token in doc if token.tag_ == "VB" or token.tag_ == "VBD" or token.tag_ == "VBG" or token.tag_ == "VBN" or token.tag_ == "VBP" or token.tag_ == "VBZ")

def adjectives_count(doc):
    return sum(1 for token in doc if token.tag_ == "JJ" or token.tag_ == "JJR" or token.tag_ == "JJS")

def adverbs_count(doc):
    return sum(1 for token in doc if token.tag_ == "RB" or token.tag_ == "RBR" or token.tag_ == "RBS")

def pronouns_count(doc):
    return sum(1 for token in doc if token.tag_ == "PRP" or token.tag_ == "PRP$" or token.tag_ == "WP" or token.tag_ == "WP$")

def conjunctions_count(doc):
    return sum(1 for token in doc if token.tag_ == "CC")

def nouns_count(doc):
    return sum(1 for token in doc if token.tag_ == "NN" or token.tag_ == "NNS" or token.tag_ == "NNP" or token.tag_ == "NNPS")

def entities_count(doc):
    return len(doc.ents)

def stop_words_count(doc):
    return sum(1 for token in doc if token.is_stop)

def punctuation_count(doc):
    return sum(1 for token in doc if token.is_punct)

def digits_count(doc):
    return sum(1 for token in doc if token.is_digit)

def avg_sentence_length(doc):
    return len(doc) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0

def noun_phrases_count(doc):
    return len(list(doc.noun_chunks))

def avg_noun_phrase_length(doc):
    return sum(len(chunk.text) for chunk in list(doc.noun_chunks)) / len(list(doc.noun_chunks)) if len(list(doc.noun_chunks)) > 0 else 0

def avg_entity_length(doc):
    return sum(len(ent.text) for ent in list(doc.ents)) / len(list(doc.ents)) if len(list(doc.ents)) > 0 else 0

def numeric_count(doc):
    return sum(1 for token in doc if token.is_digit)

def quotes_count(text):
    return text.count('"') + text.count("'")

def trigram_length(doc):
    trigrams = list(zip(doc, doc[1:], doc[2:]))
    return len(trigrams)

def bigram_length(doc):
    bigrams = list(zip(doc, doc[1:]))
    return len(bigrams)

def question_mark_count(text):
    return text.count('?')

def exclamation_count(text):
    return text.count('!')

def unique_adjectives(doc):
    unique_adj = set()
    for token in doc:
        if token.tag_ in ["JJ", "JJR", "JJS"]:
            unique_adj.add(token.text)
    return len(unique_adj)

def unique_adverbs(doc):
    unique_adv = set()
    for token in doc:
        if token.tag_ in ["RB", "RBR", "RBS"]:
            unique_adv.add(token.text)
    return len(unique_adv)

def preprocess_and_extract_features(df):
    features = []

    for _, row in df.iterrows():
        text = preprocess(row['ArticleText'])
        
        # Perform spaCy processing
        doc = nlp(text)
        
        # Extract spaCy features
        features.append({
            'Label': row.get('Bias', 'Unknown'),
            'Article_ID': row.get('ArticleID', ''), 
            'word_count': word_count(doc),
            'unique_words_count': unique_words_count(doc),
            'avg_word_length': avg_word_length(doc),
            'sentence_count': sentence_count(doc),
            'verbs_count': verbs_count(doc),
            'adjectives_count': adjectives_count(doc),
            'adverbs_count': adverbs_count(doc),
            'pronouns_count': pronouns_count(doc),
            'conjunctions_count': conjunctions_count(doc),
            'nouns_count': nouns_count(doc),
            'entities_count': entities_count(doc),
            'stop_words_count': stop_words_count(doc),
            'punctuation_count': punctuation_count(doc),
            'digits_count': digits_count(doc),
            'avg_sentence_length': avg_sentence_length(doc),
            'noun_phrases_count': noun_phrases_count(doc),
            'avg_noun_phrase_length': avg_noun_phrase_length(doc),
            'avg_entity_length': avg_entity_length(doc),
            'numeric_count': numeric_count(doc),
            'quotes_count': quotes_count(text),
            'trigram_length': trigram_length(doc),
            'bigram_length': bigram_length(doc),
            'question_mark_count': question_mark_count(text),
            'exclamation_count': exclamation_count(text),
            'unique_adjectives': unique_adjectives(doc),
            'unique_adverbs': unique_adverbs(doc),
        })

    return pd.DataFrame(features)
    
spacy_features_df = preprocess_and_extract_features(merged_df)

# Save the updated DataFrame to a new CSV file
spacy_features_df.to_csv('Hyperpartisan_Dataset.csv', index=False)