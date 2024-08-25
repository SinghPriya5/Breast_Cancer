<p align="center">
<img  width='400' height='300' src="https://github.com/SinghPriya5/Breast_Cancer/blob/main/static/images/giphy.webp"></p>
<div style="text-align: center; color: #2c3e50; font-family: 'Trebuchet MS', sans-serif;">
  <h1 style='color:#e74c3c; font-size: 3em; letter-spacing: 2px;'>✿ 𝓑𝓻𝓮𝓪𝓼𝓽 𝓒𝓪𝓷𝓬𝓮𝓻 𝓒𝓵𝓪𝓼𝓼𝓲𝓯𝓲𝓬𝓪𝓽𝓲𝓸𝓷 ✿</h1>
</div>
<img align="right" width="500" height="600" src="https://github.com/SinghPriya5/Breast_Cancer/blob/main/static/images/cancer.webp">

<h3>🌸 Table of Content</h3>

* [Problem Statement](#Problem-Statement)
* [Goal](#Goal)
* [Approach](#Approach)
* [Data Collection](#Data-Collection)
* [Project Various Step](#project-various-step)
    * [Data Visualization](#Data-Visualization)
    * [Model Training](#Model-Training)
    * [Model Evalution](#Model-Evaluation)
    * [Model Selection](#Model-Selection)
    * [Model Dump](#Model-Dump)
* [Idel used](#Idel-used)
* [Model Accuracy](#Model-Accuracy)
* [Continuous Improvement](#Continuous-Improvement)
* [Deployed](#Deployed)
* [Model Interpretation](#Model-Interpretation)
* [Web View](#Web-View)
* [Bug or Feature Request](#Bug-or-Feature-Request)
* [Future Scope](#Future-Scope)
* [Conclusion](#Conclusion)

## <h3>🌷Problem Statement:</h3>
<ul style="font-family: 'Courier New', monospace;">
  <h2>Problem Statements for Breast Cancer Classification</h2>

<ol>
    <li>
        <h3>Feature Distribution Analysis</h3>
        <p><strong>Objective:</strong> Analyze the distribution of <code>radius_mean</code>, <code>area_mean</code>, and <code>texture_mean</code> in breast cancer patients.</p>
        <p><strong>Problem:</strong> How do these features vary across the dataset, and what insights can be drawn from their distribution patterns?</p>
    </li>
    <li>
        <h3>Diagnosis and Class Imbalance</h3>
        <p><strong>Objective:</strong> Examine the distribution of diagnoses in the dataset.</p>
        <p><strong>Problem:</strong> What is the total number of diagnoses in each cancer group (benign vs malignant), and is there a significant class imbalance that could affect model training and evaluation?</p>
    </li>
    <li>
        <h3>Comparative Feature Analysis</h3>
        <p><strong>Objective:</strong> Compare key features between benign and malignant cancer groups.</p>
        <p><strong>Problem:</strong> How do the distributions of <code>smoothness_mean</code>, <code>concavity_mean</code>, and <code>concave points_mean</code> differ between these groups, and what implications do these differences have for classification?</p>
    </li>
    <li>
        <h3>Variance Analysis</h3>
        <p><strong>Objective:</strong> Identify the feature with the greatest variance in the dataset.</p>
        <p><strong>Problem:</strong> Which feature exhibits the highest variance, and how can this variance be visualized to understand outliers or the spread of data within the dataset?</p>
    </li>
    <li>
        <h3>Standard Error Analysis by Diagnosis</h3>
        <p><strong>Objective:</strong> Investigate the variation of <code>radius_se</code> between different diagnosis groups.</p>
        <p><strong>Problem:</strong> How does the total <code>radius_se</code> differ between benign and malignant groups, and what insights can be gained by visualizing this data in a pie chart?</p>
    </li>
    <li>
        <h3>Correlation Analysis</h3>
        <p><strong>Objective:</strong> Explore relationships between features in the dataset.</p>
        <p><strong>Problem:</strong> What are the correlations between different features, and are there any strongly correlated pairs that could impact the performance of predictive models?</p>
    </li>
    <li>
        <h3>Feature Interaction and Clustering</h3>
        <p><strong>Objective:</strong> Analyze how key features interact across different diagnosis groups.</p>
        <p><strong>Problem:</strong> Can pair plot visualizations reveal clusters or separability in the dataset that could be useful for improving model accuracy?</p>
    </li>
</ol>


## <h3>🌸Goal:</h3>
<div style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  The main goals of breast cancer classification using machine learning are:
  <ul>
    <li><b>Accurate Diagnosis:</b> Classify tumors accurately based on their characteristics to support medical diagnosis.</li>
    <li><b>Early Detection:</b> Enable early detection of malignant tumors, improving treatment outcomes.</li>
    <li><b>Automation:</b> Automate the classification process for quicker, more efficient diagnoses.</li>
    <li><b>Clinical Insights:</b> Understand how different attributes impact the classification to assist in medical decisions.</li>
    <li><b>Patient Guidance:</b> Provide patients with reliable diagnostic information to aid in understanding their condition.</li>
  </ul>
</div>

## <h3>🌷Approach:</h3>
<div style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  The classical machine learning tasks include Data Exploration, Data Cleaning, Feature Engineering, Model Building, and Model Testing. Various machine learning algorithms will be explored to determine the best fit for this classification task.
</div>

## <h3>🌸Data Collection:</h3>
<ul style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  <li><b>Dataset:</b> A dataset containing various tumor attributes and their corresponding diagnosis.</li>
  <li><b>Example Features:</b>
  <ul>
    <li><b>radius_mean:</b> Mean of distances from the center to points on the perimeter of the tumor.</li>
    <li><b>texture_mean:</b> Mean of the standard deviation of gray-scale values in the tumor image.</li>
    <li><b>perimeter_mean:</b> Mean size of the tumor's perimeter.</li>
    <li><b>area_mean:</b> Mean area occupied by the tumor.</li>
    <li><b>smoothness_mean:</b> Mean of the local variation in radius lengths of the tumor.</li>
    <li><b>compactness_mean:</b> Mean of the perimeter squared divided by the area minus 1.0.</li>
    <li><b>concavity_mean:</b> Mean of the severity of concave portions of the tumor's contour.</li>
    <li><b>concave points_mean:</b> Mean number of concave portions in the tumor's contour.</li>
    <li><b>symmetry_mean:</b> Mean measure of symmetry of the tumor.</li>
    <li><b>fractal_dimension_mean:</b> Mean of the "coastline approximation" minus 1, representing the fractal dimension of the tumor's contour.</li>
    <li><b>radius_se:</b> Standard error of the mean distance from the center to points on the perimeter of the tumor.</li>
    <li><b>texture_se:</b> Standard error of the standard deviation of gray-scale values in the tumor image.</li>
    <li><b>perimeter_se:</b> Standard error of the mean size of the tumor's perimeter.</li>
    <li><b>area_se:</b> Standard error of the mean area occupied by the tumor.</li>
    <li><b>smoothness_se:</b> Standard error of the local variation in radius lengths of the tumor.</li>
    <li><b>compactness_se:</b> Standard error of the perimeter squared divided by the area minus 1.0.</li>
    <li><b>concavity_se:</b> Standard error of the severity of concave portions of the tumor's contour.</li>
    <li><b>concave points_se:</b> Standard error of the number of concave portions in the tumor's contour.</li>
    <li><b>symmetry_se:</b> Standard error of the measure of symmetry of the tumor.</li>
    <li><b>fractal_dimension_se:</b> Standard error of the "coastline approximation" minus 1, representing the fractal dimension of the tumor's contour.</li>
    <li><b>radius_worst:</b> Worst (largest) mean distance from the center to points on the perimeter of the tumor.</li>
    <li><b>texture_worst:</b> Worst (largest) standard deviation of gray-scale values in the tumor image.</li>
    <li><b>perimeter_worst:</b> Worst (largest) size of the tumor's perimeter.</li>
    <li><b>area_worst:</b> Worst (largest) area occupied by the tumor.</li>
    <li><b>smoothness_worst:</b> Worst (largest) local variation in radius lengths of the tumor.</li>
    <li><b>compactness_worst:</b> Worst (largest) perimeter squared divided by the area minus 1.0.</li>
    <li><b>concavity_worst:</b> Worst (largest) severity of concave portions of the tumor's contour.</li>
    <li><b>concave points_worst:</b> Worst (largest) number of concave portions in the tumor's contour.</li>
    <li><b>symmetry_worst:</b> Worst (largest) measure of symmetry of the tumor.</li>
    <li><b>fractal_dimension_worst:</b> Worst (largest) "coastline approximation" minus 1, representing the fractal dimension of the tumor's contour.</li>
  </ul>
</li>

  <li><b>Target Variable:</b> <b>diagnosis:</b> Whether the tumor is benign or malignant.</li>
</ul>

## <h3>🌷Project Various Steps:</h3>
Data Exploration I started exploring datasets using pandas, NumPy,matplotlib and seaborn.
  
  ## <li><b>Data Visualization:</b>
  Correlation matrix and visualizations like boxplots, count plots, pair plots, scatter plots,pie chart etc., were used to understand relationships between variables.</li>
  ## <li><b>Model Training:</b>
    <ul>
      <li><b>Split Data:</b> Dataset divided into training and test sets (80% training, 20% testing).</li>
      <li><b>Model Training:</b> Models trained using the training data.</li>
      <li><b>Hyperparameter Tuning:</b> Techniques like RandomizedSearchCV used to optimize model parameters.</li>
    </ul>
  </li>
  
  ## <li><b>Model Evaluation:</b>
    Performance evaluated using metrics like Accuracy, Precision, Recall,Confusion matrix.</li>
  ## <li><b>Model Selection:</b>
    Support Vector Classifier was selected for its superior performance.</li>
  ## <li><b>Model Dump:</b>
 As per selected trained model is dumped to joblib format for app development.</li>
</ul>

## <h3>🌸Idle Used:</h3>
<ul style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  <li>Jupyter Notebook</li>
  <li>VS Code</li>
  <li>PyCharm</li>
</ul>

## <h3>🌷Model Accuracy:</h3>
<div style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  The model achieved an accuracy of 96%.
</div>

## <h3>🌸Continuous Improvement:</h3>
<ul style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  <li><b>Model Monitoring:</b> Ongoing tracking of the model’s performance to ensure accuracy.</li>
  <li><b>Retraining:</b> Periodic retraining with new data to adapt to changing medical standards.</li>
</ul>

## <h3>🌷Deployed:</h3>
Deployed on Render -- [Link](https://github.com/SinghPriya5/Breat_Cancer/issues)

<br> the instruction given on [Render Documentation](https://docs.render.com/deploy-flask) to deploy a web app.

<b>Model Deployment:</b> Deploy the model to a production environment where it can make real-time predictions.

<b>APIs:</b> Develop an API that allows users to input diamond features and receive a predicted price.

## <h3>🌸Model Interpretation</h3>
<b>Feature Importance:</b>Identify which features most influence the price prediction using techniques like feature importance scores in tree-based models or SHAP values.

##  <h3>🌷Web View:</h3>
**Frontend**
<p align="center">
  <img src="https://github.com/SinghPriya5/Breast_Cancer/blob/main/static/images/form.png" alt="Frontend" width="700" height="600">
</p>

**Inserting Value and Predicted Value**

<p align="center">
  <img src="https://github.com/SinghPriya5/Breast_Cancer/blob/main/static/images/form1.png" alt="Inserting Value" width="700" height="500"> <img src="https://github.com/SinghPriya5/Breast_Cancer/blob/main/static/images/form2.png" alt="Predicted Value" width="700" height="500">
</p>

## <h3>🌸Bug or Feature Request</h3>

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/SinghPriya5/Breast_Cancer/issues) here by including your search query and the expected result

## <h3>🌷Future Scope:</h3>
<ul style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  <li>Explore additional algorithms to enhance performance.</li>
  <li>Optimize Flask application for better performance.</li>
  <li>Enhance the frontend for improved user experience.</li>
  <li>Extend the model to include other medical insights and analysis.</li>
</ul>

## <h3>🌸Conclusion:</h3>
<div style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  Breast cancer classification using machine learning is a comprehensive approach, from data collection and preprocessing to model training, evaluation, and deployment. Accurate classification aids healthcare professionals and patients in making informed decisions, leading to improved outcomes.
</div>
<div style="text-align: center; font-family: 'Trebuchet MS', sans-serif; color: #e74c3c; margin-top: 20px;">
  <h2>Thanks!!!</h2>
</div>
