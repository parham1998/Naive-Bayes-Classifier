# Naive-Bayes-Classifier (warm-up project!)
Classification of papers with Naive Bayes approach

### dataset
Dataset consists of 6 different files, where each file represents a class (class 1 to 6). there is a Train and Test file in each of that files.
there are 100 papers in the Train file and 15 papers in the Test file for each class. (totally there are 600 papers for training and 90 papers for testing in all 6 classes.) 
### one of the Train file papers
![Screenshot (332)](https://user-images.githubusercontent.com/85555218/122393928-a70cc300-cf8a-11eb-8a1f-47bb29c7b5f6.png)

### classification method
STEP1: <br />
pre-processing of Train and Test files <br />
(at first, I preprocessed all papers in Train and Test files. preprocessing consists of removing punctuation, removing stop words, removing words under 3 letters, and change words to lowercase.) <br />
STEP 2: <br />
finding words frequency <br />
(after preprocessing, I specified the frequency of all words in all 6 classes.) <br />
STEP 3: <br /> 
finding the best attributes (see figure1) <br />
(after specifying the frequency of the words, I specified 20 best attributes (words) in each class. to do that we should subtract each word of one class from the same word in the 5 other classes then consider the 20 highest frequency words for each class.) <br />
STEP 4: <br />
combining all attributes and build a table <br />
(after specifying 20 best attributes, I built a table with 601 rows and 121 columns for Train papers, and a table with 91 rows and 121 columns for Test papers each column specifies one of the best words. (there are 6 classes and each class contains 20 best words) and the last column specifies the class number. each row indicates which of the best words each paper has.) <br />
STEP 5: <br />
implementation of Naive Bayes algorithm (see figure2) <br /> 
STEP 6: <br /> 
classification of Test and Train data <br />
(at last, I classified the data using the Naive Bayes algorithm)
    
### results
percentage of correctly classified Train data: 99.33 <br />
percentage of correctly classified Test data: 100
    
### figure1
![best](https://user-images.githubusercontent.com/85555218/122403244-78471a80-cf93-11eb-9c36-9b3b004d330a.png)

### figure2
![naive bayes](https://user-images.githubusercontent.com/85555218/122411122-be06e180-cf99-11eb-9dec-8d97f7d20521.png)
