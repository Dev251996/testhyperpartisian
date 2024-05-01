import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import shutil
import dill
from datasets import Dataset


def opendill(filename='Pickles/hyperpartisan_distilbert_classifier.dill'):
    try:
        detector_dump = open(filename, 'rb')
        return detector_dump
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return None


class HyperpartisanDistilBertClassifier:
    def __init__(self):
        self.csv_file = "label_merged_final.csv"  #Dataset goes here
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
        self.dataset = self.load_and_process_dataset()
        self.train_dataset, self.eval_dataset = self.split_dataset()
        self.trained_model = None  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        
    def preprocess(self, text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)    # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()                  # Convert to lowercase
        # Remove leading and trailing whitespace
        text = text.strip()
        # Remove brackets
        text = re.sub(r'\[.*?\]', '', text)       
        # Remove dots
        text = text.replace('.', '')
        return text
               
    def load_and_process_dataset(self):
        df = pd.read_csv(self.csv_file)
        df['PreprocessedText'] = df['ArticleText'].apply(self.preprocess)
        hyperpartisan_dataset = Dataset.from_pandas(df)
        
        def transform_labels(example):
            label = example['Bias']
            num = 0
            if label == 'least':
                num = 0
            elif label == 'right':
                num = 1
            elif label == 'left':
                num = 2
            #elif label == 'right-center':
                #num = 3
            #elif label == 'left-center':
                #num = 4
            
            return {'labels': num}
        
        def tokenize_data(example):
            return self.tokenizer(example['PreprocessedText'], padding='max_length', truncation=True)

        hyperpartisan_dataset = hyperpartisan_dataset.map(tokenize_data, batched=True)
        hyperpartisan_dataset = hyperpartisan_dataset.map(transform_labels)

        # Remove unwanted columns
        remove_columns = ['ArticleID', 'Title', 'Bias', 'ArticleText', 'Hyperpartisan', 'PreprocessedText']
        hyperpartisan_dataset = hyperpartisan_dataset.remove_columns(remove_columns)

        return hyperpartisan_dataset

    def split_dataset(self):
        return self.dataset.train_test_split(test_size=0.1)['train'], self.dataset.train_test_split(test_size=0.1)['test']
        
    def train(self, output_dir="test_trainer", num_train_epochs=5):           
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-05,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=num_train_epochs,
            seed=42,
            logging_dir='./logs',
            logging_steps=10
        )

        # Move model to GPU if available
        self.model.to(self.device)
        print(f"Using {self.device} for training.")
        
        trainer = Trainer(
            model=self.model, 
            args=training_args, 
            train_dataset=self.train_dataset, 
            eval_dataset=self.eval_dataset
        )

        trainer.train()  
        
        #Evaluate the model
        self.evaluate_model(trainer)
            
        return trainer

    def evaluate_model(self, trainer):
        # Evaluate model
        eval_result = trainer.evaluate(self.eval_dataset)
        print(eval_result)

        # Compute additional metrics
        predictions = trainer.predict(self.eval_dataset)
        pred_labels = predictions.predictions.argmax(axis=1)
        true_labels = self.eval_dataset['labels']
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        print("Confusion Matrix:")
        print(cm)
        
        # Precision, Recall, F1-Score
        print("Classification Report:")
        print(classification_report(true_labels, pred_labels))

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        
    def predict(self, input_text):
        preprocessed_text = self.preprocess(input_text)
        inputs = self.tokenizer(preprocessed_text, padding='max_length', truncation=True, return_tensors='pt')
        inputs = inputs.to(self.device)  
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        label_names = {
            0: "least",
            1: "right",
            2: "left",
            #3: "right-center",
            #4: "left-center", 
        }

        if predicted_class in label_names:
            return label_names[predicted_class]
        else:
            return "Invalid predicted label"

    def savedill(self, filename='Pickles/hyperpartisan_distilbert_classifier.dill'):
        pickle_dir = os.path.dirname(filename)
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        with open(filename, 'wb') as f:
            dill.dump(self, f)

if __name__ == "__main__":
    classifier = HyperpartisanDistilBertClassifier()  
    classifier.train()
    input_text = "*Correction appended.Just one short year ago, Texas drilling country rumbled with activity and high expectations that the good times would last a long while.So much for that.&#160;Amid a plunge in oil prices, the bonanza has paused at the very least, wiping out thousands of jobs in the oilfields and communities dependent on them.Become oneThe Texas Tribune thanks its sponsors. near $40lost nearly 340,000 jobsA barrel of West Texas crude is selling&#160; Though anything can happen in the volatile world of oil trading, all signs point to a downturn that will be longer than many expected â€“ including state revenue estimators.walloped oil-dependent communitiesBut state officials arenâ€™t sweating. The phenomenon has undoubtedly Glenn Hegarâ€œThe comptroller does not want to diminish the impact that this has had on peopleâ€™s communities and peopleâ€™s lives,â€ said Chris Bryan, a spokesman for Comptroller But statewide?&#160;â€œYouâ€™re still seeing job growth, youâ€™re still seeing revenue growth in sales tax, youâ€™re still seeing people moving here,â€ he said. â€œAnd I think that is part of that story of Texas being not just an oil and gas state anymore.â€Become oneThe Texas Tribune thanks its sponsors. Growth, but slowerThe stats back up&#160;the comptroller.&#160;The state economy is growing, though more slowly than it did amid the drilling boom.added more in each month sinceIn March, the state lost 25,400 jobs, ending a remarkable streak of 53 months of growth, according to state and federal data. Yet Texas turned around and added 33,200 jobs in May, and ending a 62-month growth streakit edged outLast June, the state collected less sales tax revenue than it did in June of 2014, Oil and mineral-related revenue makes up 10 percent of the stateâ€™s total tax collections but less than five percent of the Texas budget, according to state records.â€œThe revenue estimate is the sum of various parts, and what weâ€™re seeing with oil and gas is, itâ€™s not doing real well, but the other parts are doing very well,â€ said Dale Craymer, president of the business-backed Texas Taxpayers and Research Association and a former revenue estimator in the comptrollerâ€™s office.Wrong guesses on oil revenue â€“ sort ofBut in January, Hegarâ€™s revenue estimate pegged taxable oil prices at between $65 and $75 per barrel through 2017. Thatâ€™s far above where they are now, and far above where experts now expect them to linger, as supply stays high here and abroad and a mix of geopolitical factors â€“ including a diplomatic breakthrough that could unlock Iranâ€™s supply â€“ threaten to keep it that way.Become oneThe Texas Tribune thanks its sponsors. predictsLast week, the federal Energy Information Administration lowered its price forecasts for the coming years. The agency now almost always wrong(Texas comptrollers are Does that portend budget trouble in Texas? &#160;Not necessarily, the comptroller and other experts say. So far, oil tax revenue has come in around where Hegar predicted, largely because Texas operators are pumping more oil than expected.Wells drilled today produce far more oil than the conventionally drilled wells of years past (though the output of those wells trails off more rapidly). Operators now drill multiple wells on one site, and the rigs themselves are&#160;more efficient. Plummeting prices have only further driven operators to find new ways to eke out a profit margin.Texas oil production was still rising through May, according to the most recent figures published by the state Railroad Commission, which regulates oil and gas.â€œToday, weâ€™re producing three times the oil we did 20 years ago,â€ Craymer said.&#160; â€œWeâ€™ve got low prices on a much higher production base than weâ€™ve had historically, so that might give us a cushion as well.â€Still, experts forecast even more layoffs and declining investment in the coming months â€” resulting in lower production â€” as contracts that were signed before the prices tanked expire.That means less spending on equipment and other goods in the oil patch, taking a bite out of state sales taxes.â€œWe have had a very healthy economy and thatâ€™s help delay the impact of falling oil prices, but I think weâ€™re still going to see even slower growth ahead than what we are currently experiencing,â€ said Craymer.Next stepsWith that in mind, the comptroller's office is still keeping a close eye on oil prices, and has an opportunity to revise its forecasts in the coming months as it certifies its revenue estimate â€“ a process that takes into account the legislatureâ€™s most recent actions.&#160;â€œThis is an opportunity to look at everything again,â€ said Bryan.$6.4 billion unspentBut even if that revision proves overly optimistic, the Texas budget may still escape trouble, because lawmakers left more â€œThe question is,â€ said LeBas, â€œis the state going to stay in the black, and the answer there is yes.â€&#160;Aman Batheja contributed to this report.&#160;&#160;"
    result = classifier.predict(input_text)
    print(f"Predicted result: {result}")
    # Save the classifier object instance to a .dill file
    classifier.savedill()
