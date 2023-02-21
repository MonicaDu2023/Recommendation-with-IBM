# Recommendation-with-IBM

## Libraries
python 3.7.1 pandas numpy matplotlib sklearn pytorch

## OverView
I. Exploratory Data Analysis

Before making recommendations of any kind, I will need to explore the data. 

II. Rank Based Recommendations

To get started in building recommendations, I will first find the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users.

III. User-User Based Collaborative Filtering

In order to build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users. 

IV. Content Based Recommendations

Given the amount of content available for each article, there are a number of different ways in which someone might choose to implement a content based recommendations system. Using NLP skills, we can come up with some extremely creative ways to develop a content based recommendation system.

V. Modelling

Finally, I will implemented deep learning models such as FunkSVD and DeepFM to improve the accuracy of the recommendations and trained the models on the user-item interaction matrix.

## Acknowledgements
I wish to thank [IBM Watson Studio](https://dataplatform.cloud.ibm.com/login) platform for dataset.
