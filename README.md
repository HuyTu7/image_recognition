# image_recognition

## Structure of the folders:

```
├── src/
│   ├── knn_experiment.ipynb
│   ├── decisiontree_experiment.ipynb
|   ├── experiment.ipynb
|   ├── img_preprocessing.ipynb
├── MNIST/
├── Caltech10/
```

## Instructions: 

### Requirements:

Assuming that your system is in python 2.7 which includes these packages
+ opencv-python 
+ scikit-learn
+ matplot-lib
+ pillow
+ scipy 
+ numpy
+ jupyter 

which can be easily installed through `pip`. 

### Running Instructions:

Please make sure the folders structure are similar with data included in MNIST and Caltech10 folders for the code to run smoothly. 

- Run `python jupyter notebook` command line in the root folder 
- Run all on file `img_preprocecssing.ipynb` 
- Run all on file `knn_experiment.ipynb` for implementation and experimentation results of KNN algorithm on both datasets
- Run all on file `decisiontree_experiment.ipynb` for implementation and experimentation results of decision tree algorithm on both datasets
- Feel free to edit the call on the pickle_operating when loading the binarized version of the preprocessing dataset between with and without PCA transformation applied. 

*Notes:* 
Image preprocessing + Loading the dataset is done in the `img_preprocecssing.ipynb` file 
+ `MNIST_data_1` and `Caltech_data_1.pickle` are datasets without PCA preprocessing
+ `MNIST_data_2` and `Caltech_data_2.pickle` are datasets with PCA preprocessing


