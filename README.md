# Speech_Emotion_Recognition
Implementation of a model which classifies the voices from people of different genders and ages into 4 categories of angry, happy, sad, and neutral based on their emotions, by using different supervised and unsupervised machine learning models.

<h2> &nbsp;Supervised Learning</h2>

In supervised machine learning algorithms, a **labelled training** dataset is used first to train the underlying algorithm. This trained algorithm is then fed on the unlabeled test dataset to categorize them into similar groups.

There are several supervised learning methods. In this project we implemented the models of **SVM, Logistic Regression, KNN,** and **MLP (Neural Networks)**.

Here is a brief explanation of each of the methods named above:

<h4> &nbsp;1. Logistic Regression:</h4>

LR helps in finding the probability that a new instance belongs to a certain class. Since it is a probability, the outcome lies between 0 and 1. Therefore, to use the LR as a binary classifier, a threshold needs to be assigned to differentiate two classes.

<h4> &nbsp;2. Support Vector Machine (SVM):</h4>

Support vector machine (SVM) algorithm can classify both linear and non-linear data. It first maps each data item into an n-dimensional feature space where n is the number of features. It then identifies the hyperplane that separates the data items into two classes while maximizing the marginal distance for both classes and minimizing the classification errors.
To perform the classification, we then need to find the hyperplane that differentiates the two classes by the maximum margin.

<h4> &nbsp;3. K-Nearest Neighbor:</h4>

The K-nearest neighbor (KNN) algorithm is one of the simplest and earliest classification algorithms. The KNN algorithm does not require to consider probability values. The ‘K’ is the KNN algorithm is the number of nearest neighbors considered to take ‘vote’ from. The selection of different values for ‘K’ can generate different classification results for the same sample object.

<h4> &nbsp;4. MLP (Neural Networks):</h4>

Neural Networks are a set of machine learning algorithms which are inspired by the functioning of the neural networks of human brain. Likewise, NN algorithms can be represented as an interconnected group of nodes. The output of one node goes as input to another node for subsequent processing according to the interconnection. Nodes are normally grouped into a matrix called layer depending on the transformation they perform. Nodes and edges have weights that enable to adjust signal strengths of communication which can be amplified or weakened through repeated training. Based on the training and subsequent adaption of the matrices, node and edge weights, NNs can make a prediction for the test data.

<h4> &nbsp;Implementation of different supervised learning methods using scikit library functions in Google Colab: </h4>

In the previous parts of this project, we extracted some features from our initial dataset (voices from different genders, emotions and text IDs). Now with having different features from about 17000 samples, and labels for emotions and genders for each sample, by using the scikit library we implement each of the mentioned models above and measure the accuracy of each.

First, we defined dictionaries for 2 different genders: male, female, and 4 different emotions angry, happy, sad, and angry as below:

```ruby
Emotions = {
    1 : 'Angry',
    2 : 'Happy',
    3 : 'Sad',
    4 : 'Neutral'}

Emotions_2 = {
    'Angry' : 1,
    'Happy' : 2,
    'Sad' : 3,
    'Neutral' : 4
}

Gender = {
    'm' : 'male',
    'M' : 'male',
    'f' : 'female',
    'F' : 'female',
    'w' : 'female',
    'f ' : 'female'
}

Gender_2 = {
    'male' : 0,
    'female' : 1
}
```

Then, we used two speech feature extraction methods and combined the results to get our final set of features:

<h2> &nbsp;Feature Extraction</h2>

The first set of features were extracted using **MFCC** and **Chroma STFT** and combining the results of the 2 methods, giving us a total of 62 features (50 from MFCC and 12 from Chroma STFT).

MFCC(Mel-Frequency Cepstral Coefficients) is one of the most important methods to extract a feature of an audio signal and is used majorly whenever working on audio signals. The mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. Chroma_STFT performs short-time Fourier transform of an audio input and maps each STFT bin to chroma. In fact, it computes a chromagram from a waveform or power spectrogram.  STFT represents information about the classification of pitch and signal structure. It depicts the spike with high values in low values.

The second set of features were extracted using **librosa.feature.tempogram**. 

This feature is used when the music recording reveals significant tempo changes and the detection of locally periodic patterns becomes a challenging problem. In the concept of cyclic tempograms, the idea is to form tempo equivalence classes by identifying tempi that differ by a power of two. The resulting cyclic tempo features constitute a robust mid-level representation that reveals local tempo characteristics of music signals while being invariant to changes in the pulse level. Being
the tempo-based counterpart of the harmony-based chromagrams, cyclic tempograms are suitable for music analysis and retrieval tasks, where harmony-based criteria are not relevant. We extracted a total of 384 features in this case, and compared the results to the previous set before and after applying PCA as a method of dimension reduction on our data.

The corresponding code is as follows:

```ruby
def Extract_Data_First_Feature(data_path, voice_path):
    emotions = []
    features = []
    loaded_data = []
    Counter = 0
    datas = Read_CSV(data_path)
    datas['voice id'] = datas['voice id'].astype('str') + '.wav'
    os.chdir(voice_path)
    for file in os.listdir():           
      if file in datas['voice id'].values and file not in loaded_data:
          loaded_data.append(file)
          try:
            feature = Extract_Third_Feature(file)
            features.append(feature)
            index = datas[datas['voice id'] == os.path.basename(file)].index[0]
            emotion = Emotions[datas['emotionID'][index]]
            emotions.append(emotion)
          except Exception as e:
            print(e)
    return features, emotions
```

```ruby
def Extract_Data_Second_Feature(data_path, voice_path):
    emotions = []
    features = []
    loaded_data = []
    Counter = 0
    datas = Read_CSV(data_path)
    datas['voice id'] = datas['voice id'].astype('str') + '.wav'
    os.chdir(voice_path)
    for file in os.listdir():           
      if file in datas['voice id'].values and file not in loaded_data:
          loaded_data.append(file)
          try:
            feature = Extract_Third_Feature(file)
            features.append(feature)
            index = datas[datas['voice id'] == os.path.basename(file)].index[0]
            emotion = Emotions[datas['emotionID'][index]]
            emotions.append(emotion)
          except Exception as e:
            print(e)
    return features, emotions
```

We also implemented a shuffle algorithm to make our model more rebust:

```ruby
def Shuffle(data, labels)
```

<h4>Train-Test Split:</h4>

The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model. It is a fast and easy procedure to perform, the results of which allows us to compare the performance of machine learning algorithms for the predictive modeling problem.

In this project, after extracting the features and labels, first we split the data into test and train sets, and then implement the supervised learning methods on the split data as mentioned later in the report.

Finally, we put the features and emotions together, and split the data into train and test sets.

Here are the models for classifying the dataset based on **emotion** without applying dimension reduction to the data:

<h4>SVM:</h4>

```ruby
from sklearn.svm import SVC
kernel = ['linear', 'rbf', 'sigmoid']
gamma = ['scale', 'auto']
clf = make_pipeline(StandardScaler(), SVC(kernel, gamma))
```

<h4>KNN:</h4>

```ruby
clf = KNeighborsClassifier(weights='distance',metric='manhattan',algorithm='ball_tree',n_neighbors=i)
```

<h4>Logistic Regression:</h4>

```ruby
solver = ['saga', 'sag', 'liblinear', 'lbfgs', 'newton-cg']
max_iter = [250, 500, 100]
clf = LogisticRegression(penalty, solver, max_iter)
```

<h4>MLP:</h4>

```ruby
from sklearn.neural_network import MLPClassifier
hidden_layer_sizes = [100, 200, 300, 400, 500]
activation = ['relu', 'logistic', 'tanh']
solver = ['lbfgs', 'sgd', 'adam']
max_iter = [250, 500, 100]
clf = MLPClassifier(hidden_layer_sizes, activation, solver, max_iter)
```

Then, we find the best weights in each of the models with a **grid search algorithm**.

<h2>Evaluation metrics of the supervised learning algorithms:</h2>

<h4>1. ROC curve:</h4>

When we need to check or visualize the performance of the multi-class classification problem, we use the ROC (Receiver Operating Characteristics) curve. ROC is a probability curve and tells how much the model is capable of distinguishing between classes. The ROC curve is plotted with TPR against the FPR where TPR is on the y-axis and FPR is on the x-axis. A higher X-axis value indicates a higher number of False positives than True negatives. While a higher Y-axis value indicates a higher number of True positives than False negatives. So, the choice of the threshold depends on the ability to balance between False positives and False negatives.

<h4>2. Confusion matrix:</h4>
It is a performance measurement for machine learning classification problem where output can be two or more classes. It is a table with 4 different combinations of predicted and actual values.

It is extremely useful for measuring Recall, Precision, Specificity, Accuracy, and most importantly AUC-ROC curves.

**True Positive:** Interpretation: You predicted positive and it’s true.

**True Negative:** Interpretation: You predicted negative and it’s true.

**False Positive:** Interpretation: You predicted positive and it’s false.

**False Negative:** Interpretation: You predicted negative and it’s false.

In confusion matrices, we calculate some important values like accuracy, precision, and recall, which are explained below:

**Accuracy:** From all the classes (positive and negative), how many of them we have predicted correctly. Accuracy should be high as possible.

**Precision:** From all the classes we have predicted as positive, how many are actually positive. Precision should be high as possible.

**Recall:** From all the positive classes, how many we predicted correctly. Recall should be high as possible.

<h4>3. Learning curve:</h4>

A learning curve is a plot of model learning performance over experience or time. Learning curves are a widely used diagnostic tool in machine learning for algorithms that learn from a training dataset incrementally. Reviewing learning curves of models during training can be used to diagnose problems with learning, such as an underfit or overfit model, as well as whether the training and validation datasets are suitably representative.

Here are the **confusion matrices** for the best weights in each model:

<h4>SVM:</h4>

![My Image](images/1.png)

<h4>KNN:</h4>

![My Image](images/2.png)

<h4>Logistic Regression:</h4>

![My Image](images/3.png)

<h4>MLP:</h4>

![My Image](images/4.png)

We plot **ROC Curves** and **Learning Curves** for all 4 of the models. The results are almost the same, and look like this:

<h4>ROC:</h4>

![My Image](images/6.png)

<h4>Learning Curve:</h4>

![My Image](images/5.png)

<h2>Dimension Reduction:</h2>

The number of input variables or features for a dataset is referred to as its dimensionality. Dimensionality reduction refers to techniques that reduce the number of input variables in a dataset. More input features often make a predictive modeling task more challenging to model, more generally referred to as the curse of dimensionality. High-dimensionality statistics and dimensionality reduction techniques are often used for data visualization.

In this project, we used the PCA method for implementing the dimension reduction. The most common approach to dimensionality reduction is called principal components analysis or PCA. PCA helps us to identify patterns in data based on the correlation between features. It aims to find the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions than the original one. The orthogonal axes (principal components) of the new subspace can be interpreted as the directions of maximum variance given the constraint that the new feature axes are orthogonal to each other.

```ruby
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
pca.fit(features)
reduced_features = pca.transform(features)
```

After applying PCA on our data, we got 14 features as a result. (from both datasets, with 62 and 384 features)

By comparing the accuracy of our models both before and after applying PCA to our dataset, we saw that in some cases it helped improve the model, but in some cases the results became worse.

ROC Curve after PCA:

![My Image](images/7.png)

Learning Curve after PCA:

 ![My Image](images/8.png)

Here are the methods for classifying the dataset based on **gender**:

applying the steps above for gender classification:

```ruby
svm_best = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
svm_best.fit(x_train, y_train)

knn_best = KNeighborsClassifier(weights='distance',metric='manhattan',algorithm='ball_tree',n_neighbors=3)
knn_best.fit(x_train, y_train)

lgr_best = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=500)
lgr_best.fit(x_train, y_train)

mlp_best = MLPClassifier(hidden_layer_sizes=(300 ,), activation='logistic', solver='lbfgs', max_iter=250)
mlp_best.fit(x_train, y_train)
```

Here are the confusion matrices for the best weights in each model:

<h4>SVM:</h4>

![My Image](images/9.png)

<h4>KNN:</h4>

![My Image](images/10.png)

<h4>Logistic Regression:</h4>

![My Image](images/11.png)

<h4>MLP:</h4>

![My Image](images/12.png)

We plotted the ROC Curve for all 4 of the models. The results are almost the same. For instance, for the KNN model the results look like this:

![My Image](images/13.png)

We plotted the Learning Curve for all 4 of the models. The results are almost the same, and look like this:

![My Image](images/14.png)

Next, we applied PCA to the **gender classification** model. The ROC curve after dimension reduction:

![My Image](images/16.png)

The Learning curve after dimension reduction:

![My Image](images/15.png)

We did some more investigation regarding the **evaluation metrics**, which can be seen in the Jupyter notebook uploaded above.
