**About:** 

In recent years, Machine Learning has had a significant impact in the performance of real-world applications. This has largely been made possible through the development of deep learning models trained on the datasets, that have been carefully curated by researchers all over the world. However, getting these datasets ready for the ML models is a tedious process. This is mainly because of the two reasons: 

- Unavailability: The data may be kept private, or enough data is not available for the ML models. 
- Expensive: Getting the dataset hand-engineered for features may require an expert’s intervention. This is often the case with the medical imaging dataset, where accurate labels to body scans (for example) can only be provided by a doctor. This leads to an increased cost for the overall data gathering process. 
- Time consuming: These datasets are time-consuming to create and organise in a form ready to be fed to a model for training. 

![](CS579\_%20Final%20Report.001.png)

Image source: <http://ai.stanford.edu/blog/weak-supervision/>  

In  order  to  overcome  this  “bottleneck”  of  unlabeled  data,  weak  supervision is used. Here, instead of ground-truth labeled training set, we use: 

- Unlabeled data, or cheap and low-quality data labeled by non-experts 
- Higher level supervision over unlabeled data from the Subject Matter Experts (SMEs). This can be in the form of heuristics, constraints, expected distributions, etc. 
- Supervision from pre-trained models. 



**Applications in social media:** 

Weak  supervision  finds  its  applications  in  tasks  that  require  labeled  dataset,  but  only  an unlabeled data is available, probably along with some form of supervision. This supervision can be noisy, biased or both. In social media context, depending on the task at hand, examples of weak social supervision may include:  

- Credibility and reliability of a user or a community 
- Frequency of generated posts 
- Comments, likes, dislikes, etc., on the generated post 
- The kind of audience targeted by the author of a post 
- Relationship between different entities in a social media network 

Previously weak social supervision has been used for: 

- Detecting fake news and disinformation [\[1\]](https://arxiv.org/pdf/1910.11430) 
- Detecting cases of cyberbullying [\[2\]](https://cra.org/ccc/weakly-supervised-cyberbullying-detection-social-media/) 
- Stance Learning [\[3\]](https://www.ml.cmu.edu/research/dap-papers/f18/dap-kumar-sumeet.pdf) 



**Project Objective:** Demonstrate the application of weak social supervision to classify the intent of Brexit related tweets. 



**Dataset Information:**  

The dataset contains 

- A small sample of tweets (~800) taken from Twitter during the events of EU-Referendum in 2016 
- Out of these, only 130 (approximately) tweets are labeled. 
- The tweets are labeled with either ‘stay’ or ‘leave’, suggesting the intent of the tweet. 



**Procedure:** 

Broadly followed these steps to implement this project: 

- Generate labeling functions and train a labeling model using these labeling functions and the labeled subset (ground\_truth) of the dataset.  
- This labeling model is in turn used to generate labels (albeit noisy) for the unlabeled subset of the dataset. 
- Fine-tune  the  labeling  model/functions  as necessary to maximize the coverage, and minimize the conflicts. 
- Combine the datasets (ground\_truth and noisy labeled) to get a noisy dataset. 
- Preprocess the tweets to truncate informations such as: URLs, @mentions, #hashtags, etc. 
- Train a classifier first using only the ground\_truth labels and then the combined dataset obtained above. 
- Obtain the results and compare their performance. 



**Implementation:**  



**Libraries Used:**  

Standard  python  libraries  required  for  a  ML  program  such  as  nltk,  numpy  scipy,  pandas, sklearn, etc.  

In  addition,  I  have  used  snorkel  to  provide  weak  supervision  sources  through  Labeling Functions. The Snorkel project started at Stanford in 2016, is a platform to build training data programmatically. 

![](CS579\_%20Final%20Report.002.png)

Image source: <http://ai.stanford.edu/blog/weak-supervision/>  



**Labeling functions used:**  

- Hashtag lookup for ‘stay’ class 
- Hashtag lookup for ‘leave’ class 
- Keyword lookup for ‘stay’ class 
- Keyword lookup for ‘leave’ class 
- Possible synonyms of keywords being looked up 

Note: Keywords and Hashtags to lookup for can be further refined to improve the coverage of the labeling model. A simple approach is to do a prior Exploratory Data Analysis (EDA) on frequency of these keywords and hashtags. In this project, I have performed this for hashtags, and the most frequent ones are being looked up in the implementation. 



**Steps to run the project:** 

- All the necessary data files are present in the folder ‘data’. 
- The intermediate data files are saved in and read from from the ‘data’ folder. 
- First run ‘LabelingFunctions.py’ to generate and train a Labeling Model, and evaluate the performance measure. It also generates and saves the noisy labels for the unlabeled subset of the dataset.  
- Then, run ‘preprocess\_tweets.py’ to preprocess tweets in the dataset with ground\_truth labels and the dataset with noisy labels. 
- Finally, run ‘classification.py’ to train a logistic regression classifier using the tf-idf features of the tweets from both the datasets. In the end, it also returns the performances of the classifiers with and without weak supervision. 



**Results and Inferences:**  

Labeling functions’ performance

![](CS579\_%20Final%20Report.003.png)

![](CS579\_%20Final%20Report.004.png)

- Coverage indicates the fraction of data points, where the respective labeling function is able to label (albeit noisily) and not ‘Abstain’. 
- Overlaps indicate the fraction of data points where more than one labeling function are able to label 
- Conflicts indicate the fraction of data points where different labeling function try to label the data point differently  
- Label Model Accuracy is obtained on the subset of the dataset which is already labeled.  
- Considering that the labeling model produces noisy labels, its accuracy is expected to be low. 



Logistic Regression Classifier’s performance

![](CS579\_%20Final%20Report.005.png)

- Notice that the classifier accuracy is improved when weak supervision is used along with the ground\_truth labels. 
- This is regardless of the fact that the labeling model produces less accurate labels. 
- As a result, we saved the effort of manually labeling the dataset, while at the same time improving the accuracy of the classifier. 



**References** 

- [1] <https://arxiv.org/pdf/1910.11430>  
- [2] <https://cra.org/ccc/weakly-supervised-cyberbullying-detection-social-media/>  
- [3] <https://www.ml.cmu.edu/research/dap-papers/f18/dap-kumar-sumeet.pdf>  
- [https://www.snorkel.org](https://www.snorkel.org/)  
- <https://github.com/snorkel-team/snorkel-tutorials>  
- <http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds07.pdf>  
- <http://ai.stanford.edu/blog/weak-supervision/>  
- <http://www.cs.iit.edu/~kshu/files/ieee_intelligent.pdf>** 
