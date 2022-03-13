#  **Car aesthetics damage detection using image segmentation:**

A simple image segmentation task solved by using UNET Xception architecture. This project has below folder structure.

|Folder Name| Details|      
|----|-------|
|Model|Contains files related to model and perfomance result plots. Both .py and ipynb contains same code but in different extensions|
|data|contains training, validation and test data along with generated maks|
## **Approach and detailed walk-through of code:**

The training and validation data had only images and a JSON file (via_region_data.json).

    JSON file objects corresponding to each image. Useful object items were: 
1. **filename:** the name of the image file.
2. **regions:** an object consists of details of car damage in the image. Here a polygon mask is used and its co-ordinates were in all_points_x and all_points_y

    The coordinates extracted from this JSON file were passed to fillPoly method supplied from the OpenCV library to generate the mask (background = 0, damage location = 1) for each image in training and validation set.

After obtaining masks for all training and validation data, a TensorFlow dataset was built. The list of images and corresponding masks were organized.

Images were normalized and masks were uni-channelled using maximum over axis=3 and converted to float32. As the dataset images have random dimensions, they were resized to (128, 128) using the 'Nearest-neighbor interpolation' algorithm. NN algorithm was used because it reduces distortion while downsampling higher dimension images.

I thought to formulate this task as a binary classification for every pixel because here we don't have multiple classes we just have two (background = 0, damage location = 1). Hence it makes more sense to go with this approach. 

A state-of-art U-NET Xception architecture model was built (please check code for architecture) and compiled using:

1. **loss:** a custom loss function was built using Dice co-efficient because here we face the imbalanced class problem. The damage area loss will be muted if we don't use this loss function. Because the mask area is very less compared to the background.
2. **optimizer:** adam with learning rate = 0.001
3. **mini-batch size:** 8 (because the size of validation set and training data is very low)
4. **epoch:** 100 (Doesn't affect if it is more because we are using early stopping anyway).
5. The training data is shuffled over buffer size 100 so that the model doesn't memorize the data.
5. **metrics:** 
    * **binary-accuracy:** because we are realizing the task as a binary classification for every pixel.
    * **Dice-coefficient:** track the progress of true and predicted mask similarity.

Two callback functions for fit() function were used, listed below:
1. *DisplayIntermediateResultsCallback:* to display prediction for every 5 epochs so that the progression of training can be monitored
2. *EarlyStoppingAtMinLoss:* To stop training if the loss increase more than 5 times during the training phase.

    After training, the results were plotted for training and validation accuracy. These plots can be found in the Results folder.

    Finally, The trained model was tested by using some testing datasets (two samples). The predicted masks were displayed.

## **Requirements:**

    The following libraries were used to facilitate constructing a solution for the task.

|Library name|Version number|
|----|----|
|Python|3.8.10|
|Numpy|1.21.4|
|Tensorflow|2.7.0|
|Matplotlib|3.5.0|
|opencv|4.0.1|
|json|2.0.9|

## **Results:**
***Binary Accuracy:*** *Calculates how often predictions match binary labels.*

***Dice co-efficient:*** *similarity index between true and predicted mask*

|Phase|Binary Accuracy|Dice co-efficient|
|---|---|---|
|Training|[92%, 96%)|[0.1, 0.4]|
|Validation|[90%, 94%]|below 0.01|

Insights about the results:
* The Dice-coefficient are very poor, even the predicted mask for every 5 epochs proves it, that true mask and predicted mask are very dissimilar.

* The validation Dice-coefficient are even poorer and is way below 0.1. Hence proves that the model is getting over-fitted to some extent for training data and struggling to generalize. Hence we can see good accuracy results and still get poor results on unseen data. 
* The predicted masks on testing dataset were very poor, in fact, it was not able to even come anywhere around the true damage area. Even after trying multiple tweaking to the model hyper-parameters.

## **Discussion:**

    We saw that even though the validation accuracy was pretty good, the model performed very poorly on the test dataset. 

My argument for the poor performance and some techniques to tackle it are listed below:

1. **Batch normalization:**

    One possible reason for this kind of behaviour could be batch normalization.

    During the training phase, the batch is normalized w.r.t. mean and variance of the batch in hand.

    However, in the testing phase, the batch is normalized w.r.t. the moving average of previously observed mean and variance during training.
    
    As the size of the training dataset is too small the layer is not able to find a good approximation for mean and variance.
    
    To prove my point, I forced the model not to normalize the batch using the learnt mean and accuracy of moving statistics by passing flag training=True to the BatchNormalization layer. The predicted mask for this new configuration was better than the previous one. Even the validation and training accuracy were identically improving. The results can be seen in "Results\Predicted_mask_Training_True.png"

2. **Momentum:**
    
    As we have a very small mini-batch size (8 samples) there is high noise in each mini-batch update for mean and variance. Thus, the model could be facing ill-conditioning and converging to local minima for mean and variance due to the high value of momentum = 0.99 (default).
    
    Hence I tried giving small values like 0.01 and 0.1. These small value for momentum smoothens the updates by nullifying the vertical component of gradient vectors. The predictions were again better than the base model. The results can be seen in
        'Results\Predicted_mask_BN_0.1.png'  
        'Results\Predicted_mask_BN_0.01.png'  
    
3. **Small size dataset:**

    This is the biggest reason behind the poor learning of the model. Hence we need more data if we want the model to learn better and perform better while inferencing.

## **Conclusion:**

I initially tried with BinaryCrossentropy loss function and got some results, I thought the poor performance over test set is because of imbalanced class. Then I tried the Dice co-efficient loss function which is a State Of Art loss function for any image segmentation task. Still, I got similar results, thus I strongly believe only more data can solve this issue.

In conclusion, though the improvement techniques discussed in the Discussion section improve results slightly, it is not enough to successfully predict the damage location in the image.

Practically, Technique 1 cannot be used because we need mean and variance to be learnt during training for better performance during inference. This technique was discussed just to prove that batch normalization could be causing issues.

To fully mitigate the problem more data samples are needed because we are trying to predict for each pixel and the dimension of the input feature vector is (128 x 128 x 3).

We know that higher the dimension of the input feature vector more the training data, needed to get a well balanced predicting model. 