import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats
import json
from typing import List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics, linear_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# Load Data 

train_df = pd.read_csv('Dataset/labelled_training_data.csv')
test_df = pd.read_csv('Dataset/labelled_testing_data.csv')
validation_df = pd.read_csv('Dataset/labelled_validation_data.csv')

if train_df.columns.all() == test_df.columns.all() == validation_df.columns.all():
    print("Same")
else:
    print("Not Same")

print(train_df.dtypes)

print(train_df.head())

print(train_df.describe(include=['object', 'float', 'int']))

# Train Data 

train_df.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Train Dataset')
plt.show()

train_df.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Train Dataset')
plt.show()


# Test Data 

test_df.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Test Dataset')
plt.show()

test_df.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Test Dataset')
plt.show()

# validation Data 

validation_df.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Validation Dataset')
plt.show()

validation_df.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Validation Dataset')
plt.show()


# Are any events labelled bothsus and evil in each dataset?

# Train Dataset

print(train_df.groupby(['sus', 'evil'])[['timestamp']].count())

train_df.groupby(['sus', 'evil'])[['timestamp']].count().plot(kind='bar')
plt.show()

# Test Dataset 

print(test_df.groupby(['sus', 'evil'])[['timestamp']].count())
print(test_df.loc[(test_df['sus'] == 1) & (test_df['evil'] == 1)].shape[0])

test_df.groupby(['sus', 'evil'])[['timestamp']].count().plot(kind='bar')
plt.show()

# Validation Dataset 

print(validation_df.groupby(['sus', 'evil'])[['timestamp']].count())

validation_df.groupby(['sus', 'evil'])[['timestamp']].count().plot(kind='bar')
plt.show()


"""
From looking at these plots, it looks fairly imbalanced across the board but this is expected with a dataset like this. From reading the paper, it's kinda' the whole point! The test dataset is the only dataset that contains evil labelled events. This means that anomoly detection approaches will probably the best place to start. Something like an Auto-Encoder or One-Class SVM.

"""

# What is the correlation of features across each dataset? 

def dataset_to_corr_heatmap(dataframe, title, ax):
    corr = dataframe.corr()
    sns.heatmap(corr, ax = ax, annot=True, cmap="YlGnBu")
    ax.set_title(f'Correlation Plot for {title}')


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (15,20))
fig.tight_layout(pad=10.0)
datasets = [train_df, test_df, validation_df]
dataset_names = ['train', 'test', 'validation']
axs = [ax1, ax2, ax3]

for dataset, name, ax in zip(datasets, dataset_names, axs):
    dataset_to_corr_heatmap(dataset, name, ax)

plt.show()

"""
What does this show?¶
All three of the datasets have a heavy correlation between userid and the associated labels which feature in the dataset (sus and/or evil).
processid and threadid are highly correlated and seem to have similar correlation vaules across all three datasets. This means that they are representing pretty much the same thing and one of them could be dropped.
The correlation plots all look significantly different. This probably means its a hard problem!

"""

# Experimenting with ways to compare the datasets

# Entropy 

datasets = [train_df, test_df, validation_df]

entropy_values = []
for dataset in datasets:
    dataset_entropy_values = []
    for col in dataset.columns:
        if col == 'timestamp':
            pass
        else:
            counts = dataset[col].value_counts()
            col_entropy = stats.entropy(counts)
            dataset_entropy_values.append(col_entropy)
            
    entropy_values.append(dataset_entropy_values)

plt.boxplot(entropy_values)
plt.title('Boxplot of Entropy Values')
plt.ylabel("entropy values")
plt.xticks([0,1,2,3],labels=['','train', 'test', 'validate'])
plt.show()


# Variation

datasets = [train_df, test_df, validation_df]

variation_values = []
for dataset in datasets:
    dataset_variation_values = []
    for col in dataset.columns:
        if col == 'timestamp':
            pass
        else:
            counts = dataset[col].value_counts()
            col_variation = stats.variation(counts)
            dataset_variation_values.append(col_variation)
            
    variation_values.append(dataset_variation_values)

plt.boxplot(variation_values)
plt.title('Boxplot of Variation Values')
plt.ylabel("Variation values")
plt.xticks([0,1,2,3],labels=['','train', 'test', 'validate'])
plt.show()

"""
Exploring the Non-Numeric Cols¶
Let's use the train_df for this exploration

From looking at the annex of the paper related to this dataset, the eventName columns maps directly to the eventId column as shown in the cell below, becuase of this, it'll be ignored in the analysis. (Look at line 2 (idx 1) and line 8 (idx 7)

"""


print(train_df.loc[:, ['eventId', 'eventName']].head(10))

print(train_df.loc[:, ['processName', 'hostName', 'args']].head(10))



# Check Unique Values for all three of the fields above

def column_uniques(df, col):
    print(f'{col} - Uniques:\n\n{df[col].unique()} \n\nNo. Uniques: {df[col].nunique()}')

column_uniques(train_df, 'processName')


"""
This column contains the process names and could be processed further to create binary features. For example, a feature called amazon or not could be one or systemd or not. Something to explore later.

"""

print(column_uniques(train_df, 'hostName'))


# This column looks very useful but also very messy. Let's create a small sub-sample and investigate futher


"""
Down the Rabbit Hole - args column¶
Let's create a small sample of 15 random rows
"""

sample = train_df['args'].sample(n=15, random_state=1)
print(sample)


sample_df = pd.DataFrame(sample)
print(sample_df)

# As we can see from this sample, there seems to be a pattern here. All of the values start with [{ and end with }] which suggests it could be a dictionary within a list or maybe even json. Let's try to process a row and see if we can load it as json.

print(sample_df.iloc[0])

sample1 = sample_df.iloc[0]
sample1 = sample1.replace("[", "").replace("]", "").replace("'", '"')
sample1 = sample1[0]
print(sample1)

sample1 = json.dumps(sample1)
test1 = json.loads(sample1)
print(test1)


# This looks like a potential option if we can get each of the fields into a JSON compatible state. Let's see how it works on the rest of the sample

def strip_string(input_str):
    """
    Takes an input string and replaces specific
    puncutation marks with nothing
    
    Args:
        input_str: The string to be processed
    
    Returns:
        The processed string
    """
    assert isinstance(input_str, str)
    return input_str.replace("[", "").replace("]", "").replace("'", '"')


sample_df['stripped_args'] = sample_df['args'].apply(strip_string)


for i in sample_df['stripped_args']:
    print(i)
    print('\n')


"""
Looks like we have a pretty big problem here. Most of the records within our sample are actually multiple dictionaries/dictonary like objects with the same keys. It might be worth changing tactics and seeing if what we can do with the original field.

"""
"""
Lets take a more complicated example¶
The 3rd row in the sample is a bit more complicated. We have four sets of values which have the same keys (name, type and value) and a good variation of values.

"""

print(sample_df['args'].iloc[2])

test2 = sample_df['args'].iloc[2]

# Stage One: Isolate each individual dictionary-like object

split_test2 = test2.split('},')
split_test2

"""
Stage Two: Clean up the string by replacing punctuation and stripping blank space
Note: I have taken a similar approach to the string cleaning as the example above. The key difference here is that I am using a list comprehension to process all of the strings at once rather than just individual strings

"""

strings = [string.replace("[", "").replace("]", "").replace("{", "").replace("'", "").replace("}", "").lstrip(" ") for string in split_test2]
print(strings)

"""
We are getting there. As you can see from the above output, we now have each of the dictionary-like objects cleaned and each one is an item within a list minus all of the extra punctuation. The next stage is to break up each string to give us key:value pairs. For example, for the first string in the list above name: dirfd, type: int, value: -100, we want to get ['name: dirfd', 'type: int', 'value: -100']
Stage Three: Split each string into it's key:value pairs

"""

list_of_lists = [item.split(',') for item in strings]
print(list_of_lists)


# This is starting to look promising! The next stage is to break each of element in each list into it's key:value pairs and then turn into an actual dictionary.

# Stage Four: Breaking each element into key value pairs and convert to dictionary

output = []
for lst in list_of_lists:
    for key_value in lst:
        key, value = key_value.split(': ', 1)
        if not output or key in output[-1]:
            output.append({})
        output[-1][key] = value

print(output)


"""
Stage 5: Convert the List of Dictionaries into a Pandas Dataframe using json_normalize
This stage has two steps - The first is to dump the list of dictionaries to a JSON object and then use pd.json_normalize and json.loads to load it into a DataFrame.

"""

json_output = json.dumps(output)

interim_df = pd.json_normalize(json.loads(json_output))
print(interim_df)

# This is starting to look much better but we still have a major problem. Each row within the dataset contains the args field and as we can see from the output above, we have generated 4 rows worth of data for just one row. The next stage is to use some pandas magic to turn this dataframe into a single row.

"""
Stage 6: Convert iterim_df into a single row
Note: The 2/3 line version of this can be found in a couple of cells time, I have broken it out to make it eaiser to understand how each steps works for folks that are unfamiliar with this sort of processing

"""

"""
Step 1: Unstack
I like to think of unstack as like a whole dataframe groupby where you are grouping stuff by its column.

"""


print(interim_df.unstack())

# Step 2: Turn into a DataFrame

print(interim_df.unstack().to_frame())


# Step 3: Transpose the DataFrame (Flip it so it's horizontal instead of vertical)

print(interim_df.unstack().to_frame())

# Step 4: Sort the Indexes so each set of values is next to each other

print(interim_df.unstack().to_frame().T.sort_index(1,1))

# Stage 7: Pulling it all together and Tidy up Column Names

final_df = interim_df.unstack().to_frame().T.sort_index(1,1)
final_df.columns = final_df.columns.map('{0[0]}_{0[1]}'.format)
print(final_df)

"""
This looks exactly how we want it. All we need to do now is create a function that pull all of the stages above together and see how it shapes up on the whole dataset (very slow I bet)
args Processing Function + Test on Sample

"""


def process_args_row(row):
    """
    Takes an single value from the 'args' column
    and returns a processed dataframe row
    
    Args:
        row: A single 'args' value/row
        
    Returns:
        final_df: The processed dataframe row
    """
    row = row.split('},')
    row = [string.replace("[", "").replace("]", "").replace("{", "").replace("'", "").replace("}", "").lstrip(" ") for string in row]
    row = [item.split(',') for item in row]
    
    processed_row = []
    for lst in row:
        for key_value in lst:
            key, value = key_value.split(': ', 1)
            if not processed_row or key in processed_row[-1]:
                processed_row.append({})
            processed_row[-1][key] = value
    
    json_row = json.dumps(processed_row)
    row_df = pd.json_normalize(json.loads(json_row))
    
    final_df = row_df.unstack().to_frame().T.sort_index(1,1)
    final_df.columns = final_df.columns.map('{0[0]}_{0[1]}'.format)
    
    return final_df

"""
Let's test out the function above. First, we will convert the sample_df['args'] column to a list so we can iterate through it easily. We will then loop over each item in the list and add each processed dataframe to another list called processed_dataframes. Once complete, we will then concatanate the dataframes together to see what the output looks like!

"""

data = sample_df['args'].tolist()

processed_dataframes = []

for row in data:
    ret = process_args_row(row)
    processed_dataframes.append(ret)

processed = pd.concat(processed_dataframes).reset_index(drop=True)
processed.columns = processed.columns.str.lstrip()
print(processed)

# Merge above back into Sample

sample_df = sample_df.reset_index(drop=True)
merged_sample = pd.concat([sample_df, processed], axis=1)
print(merged_sample)

"""

Process Training Dataset
From reading the paper's annex and the code from the linked github, there are several easy processing steps that we can take. The below code has been directly copied from the github repo.

"""

# Taken from here - https://github.com/jinxmirror13/BETH_Dataset_Analysis
train_df["processId"] = train_df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
train_df["parentProcessId"] = train_df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
train_df["userId"] = train_df["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
train_df["mountNamespace"] = train_df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
train_df["eventId"] = train_df["eventId"]  # Keep eventId values (requires knowing max value)
train_df["returnValue"] = train_df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error

print(train_df.head(5))

train = train_df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]]
train_labels = train_df['sus']

print(train.head(5))

print(train_labels)

if len(train_labels) == train.shape[0]:
    print("Same")
else:
    print("Not Same")

def process_args_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Processes the `args` column within the dataset
    """
    
    processed_dataframes = []
    data = df[column_name].tolist()
    
    # Debug counter
    counter = 0
    
    for row in data:
        if row == '[]': # If there are no args
            pass
        else:
            try:
                ret = process_args_row(row)
                processed_dataframes.append(ret)
            except:
                print(f'Error Encounter: Row {counter} - {row}')

            counter+=1
        
    processed = pd.concat(processed_dataframes).reset_index(drop=True)
    processed.columns = processed.columns.str.lstrip()
    df = pd.concat([df, processed], axis=1)
    
    return df

def prepare_dataset(df: pd.DataFrame, process_args=False) -> pd.DataFrame:
    """
    Prepare the dataset by completing the standard feature engineering tasks
    """
    
    df["processId"] = train_df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    df["parentProcessId"] = train_df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    df["userId"] = train_df["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
    df["mountNamespace"] = train_df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
    df["eventId"] = train_df["eventId"]  # Keep eventId values (requires knowing max value)
    df["returnValue"] = train_df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error
    
    if process_args is True:
        df = process_args_dataframe(df, 'args')
        
    features = df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]]
    labels = df['sus']
        
    return features, labels

# Prepare the train dataset without processing args

train_no_args_feats, train_no_args_labels = prepare_dataset(train_df)
train_no_args_feats.head()
train_no_args_labels.head()

# Prepare the train dataset with processing 

"""
It looks like the args processing fuction above does not properly account for all of the different possible fields within the args field and fails to process certain rows when the key value has a value of which is a list (as shown in the snippet below). If you want to see this output yourself, uncomment the function above. I might revisit the processing once I have trained some models but at least the above processing function get's us at least half the way there.

"""

# Prepare Train, Test and Validate Datasets without processing args


# As the dataset is pre-split for us, each dataset just needs to be processed using the helper functions above.

train_df_feats, train_df_labels = prepare_dataset(train_df)
test_df_feats, test_df_labels = prepare_dataset(test_df)
val_df_feats, val_df_labels = prepare_dataset(validation_df)


# Let's Train some models

# Helper Functions

def metric_printer(y_true, y_pred):
    
    y_true[y_true == 1] = -1
    y_true[y_true == 0] = 1
    
    metric_tuple = precision_recall_fscore_support(y_true, y_pred, average="weighted", pos_label = -1)
    print(f'Precision:\t{metric_tuple[0]}')
    print(f'Recall:\t\t{metric_tuple[1]:.3f}')
    print(f'F1-Score:\t{metric_tuple[2]:.3f}')

def output_roc_plot(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Isolation Forest')
    display.plot()
    plt.show()

# Isolation Forest

clf = IsolationForest(contamination=0.1, random_state=0).fit(train_df_feats)


y_pred= clf.predict(val_df_feats)
y_probas = clf.score_samples(val_df_feats)
metric_printer(val_df_labels, y_pred)

y_pred= clf.predict(test_df_feats)
y_probas = clf.score_samples(test_df_feats)
metric_printer(test_df_labels, y_pred)

# One Class Support Vector Machine

"""
Due to the scale of this dataset, a Stohcastic Gradient Descent (SGD) version of One Class SVM is used. This is a fairly new addition to Sklearn and speeds up the training on bigger datasets by MILES.

"""

train_non_outliers = train_df_feats[train_df_labels==0]
clf = linear_model.SGDOneClassSVM(random_state=0).fit(train_non_outliers)

y_preds = clf.predict(val_df_feats)
metric_printer(val_df_labels, y_preds)

y_preds = clf.predict(test_df_feats)
metric_printer(test_df_labels, y_preds)

"""
What do these models show us?
Irrespective of the actual scores outputted, it looks like the validation set is significantly different from the train and test sets. This will make it a bit wierd for model development. Typically, you want to ensure that both the test and validation datasets are drawn from the same distrubition or you will be tweaking hyper-parameters and making model choices using the validation set which have no chance of actually performing as expected on the test set.

"""

"""
Let's get ready to train some Deep Learning Approaches
Tensor Dataloader
Before we can get to training some deep learning approaches, the first thing we need to do is get our data into a PyTorch DataLoader. The below code is not best practice but is a quick/easy way of getting the data into an PyTorch Dataloader that can be iterated over.

Stage 1: Subset the data into two dataframes - One for features and one for labels

"""

train_data = pd.DataFrame(train_df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]])
train_labels = pd.DataFrame(train_df[["sus"]])
print(train_labels)

"""
Stage 2: Convert from Pandas Dataframe to Torch Tensor
This is done by using .values which returns a numpy array version of the pandas dataframe and then conversion into a PyTorch Tensor.

Note: The dtype here is really important with other datasets. In this case, we have no float values so the choice of a 32bit integer for the dtype is OK. Typically, this is not the case and using a dtype of int32 would probably cause the data to be changed and precision lowered

"""

data_tensor = torch.as_tensor(train_data.values, dtype=torch.int32)
label_tensor = torch.as_tensor(train_labels.values, dtype=torch.int32)

# Stage 3: Combine both data_tensor and label_tensor together to make a PyTorch TensorDataset

train_dataset = TensorDataset(data_tensor, label_tensor)

# Stage 4: Create a PyTorch DataLoader using the TensorDataset
# Note: The batch size here has been chosen pretty arbitaryily.

train_data_ldr = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Stage 5: Make sure the DataLoader works properly

for i, (x, y) in enumerate(train_data_ldr):
    if i == 1:
        break
    else:
        print(f'Index: {i} \n Data Tensor: {x} \n Label Tensor: {y}')

def prepare_tensor_dataset(df: pd.DataFrame, feature_cols: List, label_col: str) -> Tuple[TensorDataset, DataLoader]:
    """
    Converts an inpurt Pandas DataFrame to a Tensor Dataset and Data Loader.
    """
    if all([col in df.columns for item in feature_cols]) and label_col in df.columns:
        
        labels = pd.DataFrame(df[[label_col]])
        features = pd.DataFrame(df[feature_cols])

        data_tensor = torch.as_tensor(features.values, dtype=torch.int32)
        label_tensor = torch.as_tensor(train_labels.values, dtype=torch.int32)
        
        tensor_dataset = TensorDataset(data_tensor, label_tensor)
        
        data_ldr = train_data_ldr = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        return tensor_dataset, data_ldr
    else:
        raise ValueError('Unable to find all columns')

label_col = 'sus'
feat_cols = ["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]

train_tensor_dataset, train_data_ldr = prepare_tensor_dataset(train_df, feat_cols, label_col)

