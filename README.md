# FusionModel


#### Code for training:
Customer 3D model: A custom 3D neural network model, consisting of 3D convolutions forming a tailored architecture. The model is a fusion of three 3D convolutional models with varying depths at different nodes.

multi_task: Multi-task model (Convolution Model and XGBoost Model trained together).

gmask_extract: Extracts the region of interest (ROI) from the entire image based on the mask information corresponding to the 3D image.

table_merge_models: XGBoost Model and Fusion Model (including hyperparameter optimization). The XGBoost Model uses an embedding approach for categorical data and bucketing for continuous data. The Fusion Model can simultaneously capture multi-scale information from medical images and clinical data.


#### Code for inference:
deepmodel_inference.py: Convolution Model prediction inference code, which loads deep learning model files and test data for prediction inference;

tabnet_inference.py: XGBoost Model prediction inference code, which loads machine learning model files and test data for prediction inference;

merge_inference_blend.py: Fusion Model prediction inference code, which simultaneously loads the test data corresponding to the Convolution Model and XGBoost Model's prediction inference results, and generates the Fusion Model prediction results.


#### Usage Instructions:
Deep Learning Model: Data should be organized in the format required by Keras, with folders serving as labels.

Load the data and execute the corresponding inference prediction code (python xxxx.py) to obtain the respective results.

The output file "result_deep.csv" contains the prediction results from the Convolutional Model.
The output file "result_table.csv" contains the prediction results from the XGBoost Model.
The output file "result_merge.csv" contains the prediction results from the Fusion Model.

