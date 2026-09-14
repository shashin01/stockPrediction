This is project is created with the goal of testing a few different ML 
algorithms for stock prediction. 

The following models have been tested:
1. LSTM

Further details on accuracy of different models and configurations can be 
found in 'findings.txt'

Usage details:
1. Run 'train.py' and provide necessary configurations
2. Training the model for a specific ticker will save model, metadata and 
sacler to 'models/TICKER/*'
3. Run 'predict.py' and provide necessary configurations

INVESTIGATION NOTES:
1) Originally, I was trying to improve the accuracy of my model by running evaluation scripts 
but I realised that I need to visualize loss over multiple epochs. 
Looking at the actual prediction values shows potential for overfitting as well.
Modified prediction script to see how loss is changing across epochs.
Modification confirms overfitting, reducing the number of epochs was the best next step. 25 -> 5 epochs