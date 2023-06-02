## SecureSense: A Data-Driven Framework for Phishing Attack Prevention

### UIC Engineering Expo 2023 Best in Show
![image](https://github.com/khuynh22/Phishing-Detection/assets/57774658/f2ff0fd3-2be1-40e7-a07d-6ea39127d978)


#### 1. Introduction:
<ul>
  <li>This is my Bachelors of Science Degree Capstone Project, where I apply my the theory of my Computer Science - Machine Learning Major and Business Analytics Minors</li>
  <li>As the rising trend of Phishing Attacks within the scope of UIC Student, I decided to build a data frames of machine learnings models with the hope to use technology to punish technology criminals!/li>
  <li>There are three ML models has been accomplished, and the website are on the process of building into production. I am planning to release the project within 2024.</li>
</ul>

#### 2. Methodology:
The models was build based on the dataset Phishing Legitimate Full.csv from Mendeley Data, with 10,000 data points from 5000 legitimate webpages and 5000 phishing webpages with 48 websites features to analyze

The project use three main Machine Learning models to build, including:
<ul>
  <li>Decision Tree Model</li>
  <li>Logistics Regression Model</li>
  <li>Random Forest Classification Model</li>
</ul>

The project also used other concepts including Mutual Infos, Spearman Coefficient, Gini Index, etc. in addition to the ML models.

Please encounter the project report to learn more about why and how these concept are implemented within the scope of this project.

#### 3. Result:
After train the model and test it using the database, here are the result of the model:

![image](https://github.com/khuynh22/Phishing-Detection/assets/57774658/d77f0831-ea7b-4248-aa43-abed2da63270)

Briefly saying, all three models provides a great outcomes, with the best model (in term of numbers from Random Forest). However, based on the reality of running the models, as well as the theoretical point of Random Forest Model (which basically run multiples of Decision Tree in runtime). Therefore, in further step of putting the model into production, Decision Tree Model could be considered for it efficient in runtime.

#### 4. Further Steps:
I am working with two other members of my team in other to develop the website where the user can input the webpage URL, and we can use NLP Model and Web Scraping to transform all the necessary features into our ML Model. After that, the data point can be process through our ML Model and output the result.

Our team are at the processed of developing the wireframe using Figma, and hopefully can make this project into production within 2024.

#### 5. Acknowledgements
This work has been conducted under the supervising of Professor Mitchell Theys and feedback from Professor Xinhua Zhang from the Department of Computer Science at UIC.
