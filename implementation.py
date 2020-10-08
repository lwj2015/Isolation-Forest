import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# prepation******************************************************************
def select_feature(data):
    return np.random.choice(data.columns)

def select_value(data,feat):
    mini = data[feat].min()
    maxi = data[feat].max()
    return (maxi-mini)*np.random.random()+mini

def split_data(data, split_column, split_value):
    data_below = data[data[split_column] <  split_value]
    data_above = data[data[split_column] >= split_value]

    return data_below, data_above

def classify_data(data):
    data = np.array(data)
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    # 找到每一个数值中最大的，其实也可以用unique_classes[0]
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification

# key functions**************************************************************
def isolation_tree(data, counter=0, max_depth=50):
    # End Loop if max depth or isolated
    if (counter >= max_depth) or data.shape[0] <= 1:
        classification = classify_data(data)
        return classification

    else:
        # Select feature
        split_column = select_feature(data)
        # Select value
        split_value = select_value(data, split_column)
        # Split data
        data_left, data_right = split_data(data, split_column, split_value)

        # instantiate sub-tree
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}

        # Recursive part
        left_answer = isolation_tree(data_left , counter+1, max_depth=max_depth)
        right_answer = isolation_tree(data_right, counter+1, max_depth=max_depth)

        if left_answer == right_answer:
            sub_tree = left_answer
        else:
            # quesiton相当于这一层的root，分别放left和right
            sub_tree[question].append(left_answer)
            sub_tree[question].append(right_answer)

        return sub_tree

def isolation_forest(df, n_trees=5, max_depth=5, subspace=256):
    forest = []
    max_depth = np.ceil(np.log2(subspace))

    for i in range(n_trees):
        # Sample the subspace
        if subspace <= 1:
            df = df.sample(frac = subspace)
        else:
            df = df.sample(n = subspace)

        tree = isolation_tree(df, 0, max_depth)
        forest.append(tree)

    return forest

def pathLength(example, iTree, path=0):
    # Initialize question
    question = list(iTree.keys())[0]
    feature_name, _, value = question.split()

    # ask question
    if example[feature_name].values <= float(value):
        answer = iTree[question][0]
    else:
        answer = iTree[question][1]

    # base case,叶结点了
    if not isinstance(answer, dict):
        return path+1
        #T_size = len(list(iTree.keys())[0])
        #return np.e + 2*(np.log(T_size-1)+0.5772156649) -2*(T_size-1)/T_size

    # recursive part
    else:
        residual_tree = answer
        return pathLength(example, residual_tree, path+1)

# Evaluate one instance
def evaluate_instance(instance,forest):
    paths = []
    for tree in forest:
        paths.append(pathLength(instance,tree))
    return paths

def predict(dataset,forest,ratio = 0.1):
    normal = []
    abnormal = []
    pred = []
    columns = dataset.columns.values
    dataset = np.array(dataset)

    for i in dataset:
        instance = pd.DataFrame(i.reshape(-1, 2), columns=columns)
        length = evaluate_instance(instance, iForest)
        pred.append(np.mean(length))

    sorted = np.sort(pred)
    split = sorted[int(len(sorted) * ratio)]

    for i in range(len(pred)):
        if pred[i] < split:
            abnormal.append(dataset[i])
        else:
            normal.append(dataset[i])

    return normal,abnormal

mean = [0, 0]
cov = [[1, 0], [0, 1]]
Nobjs = 2000
x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
#Add manual outlier
x[0]=3.3; y[0]=3.3

dataset = np.array([x, y]).T
dataset = pd.DataFrame(dataset, columns=['feat1', 'feat2'])

iForest = isolation_forest(dataset, n_trees=30, max_depth=100, subspace=256)

normal,abnormal = predict(dataset,iForest,0.15)



# visualization

plt.figure(figsize=(7,7))
axes = plt.subplot(111)
axes.scatter(x,y, s=10, color='blue')
plt.show()


axes = plt.subplot(111)

normal = np.array(normal)
abnormal = np.array(abnormal)
type1 = axes.scatter(normal[ : , 0],   normal[ : , 1],   s=10, color='blue')
type2 = axes.scatter(abnormal[ : , 0], abnormal[ : , 1], s=12, color='red')

axes.legend((type1, type2), ("norm", "anomaly") , prop={'size':12})

plt.title("Isolution Forest")
plt.show()







