# Hyperpartisan Text Classification 

# Supervised Detector 

Step 1 : Data Cleaning and Preprocessing

Script: PyMLHyperpartisanDataCleaning.py

Usage: python PyMLHyperpartisanDataCleaning.py

# #### Download the XML Dataest from the link:  https://zenodo.org/records/1489920
# I have used the below dataset files from the link for the supervised classification
# 1. articles-training-bypublisher-20181122.zip
# 2. ground-truth-training-bypublisher-20181122.zip 
# 
# Refer the two .xsd files for headers - article.xsd and ground-truth.xsd for converting XML to CSV 
# 
# Change all the file path variables with the local storage path of the XML files accordingly.
# I have renamed articles-training-bypublisher-20181122.xml to "Article.xml" and ground-truth-training-bypublisher-20181122.xml to "Groundtruth.xml"


Purpose of this code: 
1. It converts the XML files to CSV and store it in local system.
2. It cleans the data by checking null values, non-numeric values, data type conversion, drops unwanted columns and merges Article and Groundtruth files to get the labels -'BIAS'.
	Intermediate File name: Merged_march2.csv
3. Combines the labels left-center to left and right-centre to right for better performance.
	Intermediate File name: label_merged_final.csv
4. The intermediate files are stored as csv's in local system for future reference.
5. It Preprocesses and gathers text features from article text and stores it as the Final dataset file for training the model. 
	File name: Hyperpartisan_Dataset.csv

All the files are uploaded to google drive - https://drive.google.com/drive/folders/1K5nE9Yarn8audfcZwoYYYSsnlERvWDQI?usp=sharing


Step 2: Model Training and prediction

Script: main.py, MLHyperpartisanDetector.py

Dataset is too big (2GB) and the above python file(PyMLHyperpartisanDataCleaning.py) takes long time to run. 
To skip that part, you can download "Hyperpartisan_Dataset.csv" from github and start to train the detector.

Usage: python main.py

Place the "Hyperpartisan_Dataset.csv" in the local path and execute the script.


Result: 

Prediction Result: {'RandomForest': 'left,0.04,0.62,0.34'}


# Deep Learning Detector 

Dataset - Download "label_merged_final.csv" from  https://drive.google.com/drive/folders/1K5nE9Yarn8audfcZwoYYYSsnlERvWDQI?usp=sharing and 
place it in your local file path.

Script: PyDLHyperpartisan.py

Usage: python PyDLHyperpartisan.py

By Default, Detector classifies the text into three labels - left, right and least. 

If you want to classify text into five labels - least, left, right, left-center and right-center,
download "Merged_march2.csv" from the drive link, update the file path in code and execute it. Also, change the lines in code as below:

Line 16: Change the dataset file path
Line 18: Change the label number to 5 (num_labels = 5)
Line 51 to 54: Uncomment 
Line 142, 143: Uncomment

Training time: 6 hours

Total Articles: 599,334

Result:

Predicted result: left







