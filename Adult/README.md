
data_adult.py: data_preprocessing file called in main file
fair_convex.py: covariance based model

run as python3 fair_covariance.py

in data_adult.py - give adult.csv as input. 
in compute cost function change the values of lambda which are multipliers to the cost function.
You may change learning rate to get the best value. 

fair_model.py introduces the fairness constraints directly in neural network model.
run as:
python3 fair_model.py

change the values of multipliers in cost function to obtain a tradeoff between fairness and accuracy. start from 0.01 to 100
