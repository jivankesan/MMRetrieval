import os

# Directory where you want to save the model
save_model = 'weights/'

# Check if the directory exists, if not, create it
if not os.path.exists(save_model):
    os.makedirs(save_model)

