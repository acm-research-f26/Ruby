# How to use this testing data 
1. Use the sim_configuration.ipynb to generate the dataset - it was too big to fit on github so I couldn't upload all the images. The images shouldn't be more than 130 MB, let me know if you run into storage problems. 
2. Run your model on the this data set. Manifest.csv has the information foramtted in a way similar to the Euclid dataset. It has the file id,label,grade,notes,ra,dec,path,split.

# Notes/Problems you may run into 
1. For bettter retention I used .npy, so if your model uses .png you will have to convert to png.
2. If your model requires more information than just the image, then there is another csv file with more information-- labels.csv
3. To verify that this set was well distributed accross paramters, I cross verified with a Gradient Boost to see what parameters of the images a mdoel could potentially get biased from. In otherwords the validation portion of sim_configuration.ipynb takes a dataset, trains a Gradient Boosting model using 5-fold cross-validation to prevent cheating, gets the prediction probabilities, and returns a single AUROC accuracy score rounded to 3 decimal places.

In gravitational lensing or astronomical datasets, positive cases (like compound systems) often occupy a very specific range of physical parameters (e.g., they tend to be larger or have a specific Einstein radius). If your negative samples have a completely different distribution for theta_E_outer, a machine learning model will quickly figure out that it can just cheat by looking at the size (theta_E_outer) instead of learning the actual complex features. I only ran it for the radius because it's more likely to be a dominat feature because it varies so much. I fyou have time, feel free to cross reference the other features. 
