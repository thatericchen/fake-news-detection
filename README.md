# Fake News Detection
## Introduction
Fake news is a severe issue in today’s digital world, contributing to misinformation and distrust in society. Since news is now predominantly online, social media platforms have been prime targets for fake content due to their wide reach [\[1\]](https://www.researchgate.net/publication/369273557_Fake_news_detection_a_systematic_literature_review_of_machine_learning_algorithms_and_datasets). In 2018, researchers warned that “most people in mature economies will consume more false information than true” by 2022 [\[4\]](https://scholar.smu.edu/datasciencereview/vol1/iss3/10/), highlighting the urgency of addressing this issue. Recent studies show promise with machine learning classifiers such as Random Forests and XGBoost, which focus on source credibility and feature selection [\[2\]](https://ieeexplore.ieee.org/abstract/document/8709925), as well as Naïve Bayes models that have shown effectiveness in detecting fake content on platforms like Facebook [\[3\]](https://ieeexplore.ieee.org/abstract/document/8546944). Despite a few successes, existing solutions have struggled because of the sophistication of fake content, often mimicking real news in tone and structure, making it harder to detect.

## Problem Definition
Fake news threatens the integrity of digital information by manipulating public opinion, influencing political decisions, and inciting social unrest. “Fake news detection on social media presents unique characteristics and challenges that make existing detection algorithms from traditional news media ineffective or not applicable,” particularly due to the massive scale and complexity of online platforms [\[5\]](https://arxiv.org/abs/1708.01967). Compounding this difficulty is the prevalence of fake news that contains partial truths, which can confuse both readers and detection models [\[6\]](https://arxiv.org/abs/1811.00770).

To address this, our project aims to develop an automated machine learning system capable of distinguishing fake news from real news based solely on textual content. Our approach leverages [curated datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/code) of over 24,000 labeled articles—split into two sets: one for real news and one for fake. Each article includes a title and full text, which will be the primary focus of our analysis. We plan to explore the distribution of article lengths, visualize common word usage, and identify outliers such as unusually short or long articles to better understand the data prior to model development. By focusing on natural language patterns and content-based features, we aim to build a scalable and reliable solution that can combat misinformation online.

## Methods
Data Preprocessing
- We utilize NLTK for text cleaning and tokenization, removing punctuation, converting text to lowercase, and filtering out stop words. Then, we apply scikit-learn’s TfidfVectorizer to transform the cleaned text into a numerical TF‑IDF feature matrix.
- Why chosen: This pipeline ensures that only the most informative words (i.e., those that distinguish articles) are retained, reduces noise, and produces a compact representation suitable for a variety of classifiers.

Logistic Regression
- We train an unregularized logistic regression model via gradient descent to learn optimal feature weights. The model outputs a probability score for each article being fake or real and classifies based on a threshold.
- Why chosen: Logistic regression is computationally efficient, highly interpretable, and serves as a strong baseline. Its linear decision boundary and probability outputs help us understand feature importance and model confidence.

Multinomial Naïve Bayes
- We apply MultinomialNB on the TF‑IDF feature matrix, estimating class-conditional word probabilities and using Bayes’ theorem to compute the posterior likelihood of each class.
- Why chosen: Multinomial Naive Bayes is very effective for sparse, high-dimensional text representations. Its independence assumption simplifies computation, allowing fast training and inference while maintaining strong performance on word-frequency features.

Random Forest
- We implement a RandomForestClassifier, constructing an ensemble of decision trees trained on bootstrap samples of the TF‑IDF features. Predictions are made by majority vote across all trees.
- Why chosen: Random Forests capture complex, non-linear relationships between textual features without requiring extensive feature engineering. The ensemble approach reduces overfitting and improves generalization, making it robust for high-dimensional text data.

## Results and Discussion

### Logistic Regression
#### Visualizations
![logi1]('/Users/echen/GT/Spring 2025/fake-news-detection/logistic_regression.png')
![logi2](https://github.gatech.edu/ML4641Team19/CS4641-Project/assets/64872/519b136e-23a8-469e-a306-7553ca8b002c)


#### Quantitative Metrics
- F1-Real: 0.85 - 0.90
- F1-Fake: 0.80 - 0.85
- PR-AUC: 0.98
- ROC-AUC: 0.98
- Balanced Accuracy: 0.8611

#### Analysis
The model was able to achieve the goal of ≥ 0.85 for the F1-Real score and balanced accuracy, as well as the goal of ≥ 0.80 for PR-AUC.

- A F1-Real score of 0.85 - 0.90 indicates high accuracy in identifying real news.
- A F1-Fake score of 0.80 - 0.85, although lower than the goal of ≥ 0.85, still demonstrates a decent ability to identify fake news.
- A PR-AUC and ROC-AUC of 0.98 suggests that the model is effective at distinguishing between real and fake news with minimal uncertainty.
- A balanced accuracy of 0.8611 indicates strong predicative performance across both real and fake news.

Confusion Matrix: 
- True Positive (True/Predicted Label = 1): 3064
- True Negative (True/Predicted Label = 0): 4669
- False Positive (True Label = 0, Predicted Label = 1): 50
- False Negative (True Label = 1, Predicted Label = 0): 1197

Fake news were correctly identified as fake in 3064 cases while real news were correctly identified as real in 4669 cases. False positive is relatively low compared to the total instances; however, there is significantly high number of false negative, which lowers recall score. 

The model performed well because logistic regression is effective for binary classification tasks, such as fake and real news detection. This is because a binary logistic regression model performs best when the data exhibit clear patterns and linearly separate decision boundaries, which is the case with fake and real news. Fake news often contains features like exaggerated language, which can be distinguished from real news, allowing the logistic regression model to learn and classify them effectively.

### Naïve Bayes
#### Visualizations
![nb1](https://github.gatech.edu/ML4641Team19/CS4641-Project/assets/64872/84ac7145-e6bc-430b-ad1e-4dd93d3637bb)
![nb2](https://github.gatech.edu/ML4641Team19/CS4641-Project/assets/64872/a37bc6a5-6eda-4760-9a01-97050eda11c5)

#### Quantitative Metrics
- Precision: 0.9320
- Recall: 0.9357
- F1: 0.9338
- PR-AUC: 0.98
- ROC-AUC: 0.98
- Accuracy: 0.9371

#### Analysis
The model was able to achieve the goal of ≥ 0.85 for the F1 score and accuracy, as well as the goal of ≥ 0.80 for PR-AUC.

- A F1 score of 0.9338 indicates a strong balance between precision and recall. It is good at detecting fake news while avoiding incorrectly identifying real news as fake. 
- A PR-AUC and ROC-AUC of 0.98 suggests that the model is effective at distinguishing between real and fake news with minimal uncertainty.
- An accuracy of 0.9371 indicates strong predicative performance across both real and fake news.

Confusion Matrix: 
- True Positive (True/Predicted Label = 1): 3987
- True Negative (True/Predicted Label = 0): 4428
- False Positive (True Label = 0, Predicted Label = 1): 291
- False Negative (True Label = 1, Predicted Label = 0): 274

Fake news were correctly identified as fake in 3987 cases while real news were correctly identified as real in 4428 cases. The errors (false positive and false negative) are relatively low compared to the total instances, indicating high precision and recall score.

The Naïve Bayes model performed well because it is effective at analyzing high-dimensional, sparse feature spaces like those produced by TF-IDF. TF-IDF simplifies language patterns through word frequency and importance, and it works well in real and fake news detection due to the clear differences in language use between the two. The IF-IDF representation helps emphasize distinctive terms in a way that makes the classes linearly separable, allowing Naïve Bayes to separate them cleanly.

### Random Forest
#### Visualizations
![rf1](https://github.gatech.edu/ML4641Team19/CS4641-Project/assets/64872/fc315e8a-9edf-49e7-9d00-c4f545b0ba0c)
![randomforest2](https://github.gatech.edu/ML4641Team19/CS4641-Project/assets/64872/2a0e67af-cc81-402d-842d-2f4986b613b2)

#### Quantitative Metrics
- Precision: 0.9908
- Recall: 0.9350
- F1: 0.9621
- PR-AUC: 1.00
- ROC-AUC: 1.00
- Accuracy: 0.9650

#### Analysis
The model was able to achieve the goal of ≥ 0.85 for the F1 score and accuracy.

- A F1 score of 0.9621 indicates a strong balance between precision and recall. It is good at detecting fake news while avoiding incorrectly identifying real news as fake.
- A PR-AUC and ROC-AUC of 1.00 suggests that the model is effective at distinguishing between real and fake news with no uncertainty.
- An accuracy of 0.9650 indicates strong predicative performance across both real and fake news.

Confusion Matrix: 
- True Positive (True/Predicted Label = 1): 4018
- True Negative (True/Predicted Label = 0): 4666
- False Positive (True Label = 0, Predicted Label = 1): 53
- False Negative (True Label = 1, Predicted Label = 0): 243

Fake news were correctly identified as fake in 4018 cases while real news were correctly identified as real in 4666 cases. The errors (false positive and false negative) are relatively low compared to the total instances, indicating high precision and recall score. 

The Random Forest model performed well because it is effective at capturing complex relationships between language features present in news articles. TF-IDF creates a high-dimensional feature space, and Random Forest model can handle this effectively by aggregating the decisions of many individual decision trees. This ensemble approach reduces overfitting and improves generalization, allowing the model to detect complex textual patterns.

## Comparison

| Name               | Accuracy | F1       | PR-AUC | ROC-AUC |
|--------------------|----------|----------|--------|---------|
| Logistic Regression| 0.8611   | 0.80–0.90| 0.98   | 0.98    |
| Naïve Bayes        | 0.9371   | 0.9338   | 0.98   | 0.98    |
| Random Forest      | 0.9650   | 0.9621   | 1.00   | 1.00    |

### Logistic Regression
   - Strengths: Effective for binary classification tasks when the data is linearly separable.
   - Limitations: Struggles with complex, non-linear feature interactions, especially if the decision boundary is not linear. 
   - Trade-offs: Offers high interpretability and simplicity but lacks the flexibility to capture more complex pattern in the data.

### Naïve Bayes
   - Strengths: Well-suited for text data with TF-IDF features. Handles high-dimensional, sparse inputs efficiently due to the assumption of feature independence. Fast to train and easy to implement. 
   - Limitations: Assumes conditional independence which is not always the case in real language pattern, causing the model to be less flexible in capturing feature interactions compared to some model. 
   - Trade-offs: Offers fast training speed and high performance for text classification but at the cost of some flexibility in capturing complex patterns.

### Random Forest
   - Strengths: Captures non-linear relationships and complex interactions between features that are missed by other two models. Robust to overfitting due to the ensemble approach.
   - Limitations: Higher computational cost and longer training time. 
   - Trade-offs: Offers high accuracy and generalization ability at the cost of increased complexity and computational cost.

## Next Steps
After implementing and evaluating Logistic Regression, Naïve Bayes, and Random Forest, we are confident that our models are achieving strong performance in fake news detection. We have successfully met our goal for quantitative metrics, and compared to our initial logistic regression model, both Naïve Bayes and Random Forest demonstrated improved capability in accurately classifying fake news.

However, here are a few potential improvements we could explore.
- Further fine-tune hyperparameters and adjust the classification threshold to see if performance can be further improved.
- Explore XGBoost (XGBCLassifier) model. XGBoost's gradient boosting approach may offer incremental gains in performance by more effectively capturing complex patterns in the data.

## References
[1]  S. I. Manzoor, J. Singla, and Nikita, “Fake news detection using machine learning approaches: A systematic review,” Proc. 2019 IEEE 10th Int. Conf. Comput., Commun. Netw. Technol. (ICCCNT), 2019, pp. 1–6. Available: https://ieeexplore.ieee.org/abstract/document/8862770. [Accessed: Feb. 20, 2025].

[2]  J. C. S. Reis,  A. Correia, F. Murai, A. Veloso, and F. Benevenuto, “Supervised learning for fake news detection,” Proc. 2019 IEEE Int. Conf. Artif. Intell. Knowl. Eng. (AIKE), 2019, pp. 1–8. Available: https://ieeexplore.ieee.org/abstract/document/8709925. [Accessed: Feb. 20, 2025].

[3]  A. Jain and A. Kasbe, “Fake news detection,” Proc. 2018 Int. Conf. Adv. Comput. Commun. Control Netw. (ICACCCN), 2018, pp. 1–5. Available: https://ieeexplore.ieee.org/abstract/document/8546944. [Accessed: Feb. 20, 2025].

[4]  A. Thota, P. Tilak, S. Ahluwalia, and N. Lohia, “Fake news detection: A deep learning approach,” SMU Data Sci. Rev., vol. 1, no. 3, 2018. Available: https://scholar.smu.edu/datasciencereview/vol1/iss3/10/. [Accessed: Feb. 20, 2025].

[5]  K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, “Fake news detection on social media: A data mining perspective,” arXiv preprint, 2017. Available: https://arxiv.org/pdf/1708.01967. [Accessed: Feb. 20, 2025].

[6]  R. Oshikawwa, J. Qian, and W. Y. Wang, “A survey on natural language processing for fake news detection,” arXiv preprint, 2018. Available: (https://arxiv.org/abs/1811.00770). [Accessed: Feb. 20, 2025].
