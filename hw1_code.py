from itertools import count
from typing import List, Optional
from numpy.lib.function_base import select
from numpy.lib.type_check import real
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import graphviz
import math

REAL = "real"
FAKE = "fake"

def read_file(file_name: str) -> List[str]:
    file = open(file_name, "r")
    data = [line.strip() for line in file]
    file.close()
    return data

def load_data():
    headlines = read_file("clean_real.txt").copy()
    labels = [REAL] * len(headlines)
    headlines.extend(read_file("clean_fake.txt").copy())
    labels.extend([FAKE] * (len(headlines) - len(labels)))

    # now we want to split the data into training, test, and validation
    x_train, x_test, y_train, y_test = train_test_split(headlines, labels, train_size=0.7)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, train_size=0.5)

    # now vectorize the data
    vectorizer = CountVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_val_vectorized = vectorizer.transform(x_validation)

    return x_train_vectorized, x_val_vectorized, y_train, y_validation, vectorizer

def accuracy_calculator(y_true, y_pred) -> float:
    '''
    y_true - the correct set of labels/outputs
    y_pred - the predicted set of labels/outputs
    
    This method compares y_true with y_pred and returns a float score indicating how closely the two match up

    Precondition: len(y_true) == len(y_pred)
    '''
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred are inconsistent lengths")
    
    total = len(y_true)
    score = 0
    for i in range(total):
        if y_true[i] == y_pred[i]:
            score += 1
    
    return score/total


def select_data() -> DecisionTreeClassifier:
    x_train, x_val, y_train, y_val, vectorizer = load_data()

    tree_to_accuracy = {}   # maps decision trees to their accuracy scores

    # max_depth = 3, split_criteria = information gain
    t1 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    t1 = t1.fit(x_train, y_train)
    labels_predicted = t1.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T1:", accuracy)
    tree_to_accuracy[t1] = accuracy

    # max_depth = 3, split criteria = gini
    t2 = DecisionTreeClassifier(criterion="gini", max_depth=3)
    t2 = t2.fit(x_train, y_train)
    labels_predicted = t2.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T2:", accuracy)
    tree_to_accuracy[t2] = accuracy

    # max depth = 5, split criteria = entropy
    t3 = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t3 = t3.fit(x_train, y_train)
    labels_predicted = t3.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T3:", accuracy)
    tree_to_accuracy[t3] = accuracy

    # max depth = 5, split criteria = gini
    t4 = DecisionTreeClassifier(criterion="gini", max_depth=5)
    t4 = t4.fit(x_train, y_train)
    labels_predicted = t4.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T4:", accuracy)
    tree_to_accuracy[t4] = accuracy

    # max depth = 10, split criteria = entropy
    t5 = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    t5 = t5.fit(x_train, y_train)
    labels_predicted = t5.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T5:", accuracy)
    tree_to_accuracy[t5] = accuracy

    # max depth = 10, split criteria = gini
    t6 = DecisionTreeClassifier(criterion="gini", max_depth=10)
    t6 = t6.fit(x_train, y_train)
    labels_predicted = t6.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T6:", accuracy)
    tree_to_accuracy[t6] = accuracy

    # max depth = 15, split criteria = entropy
    t7 = DecisionTreeClassifier(criterion="entropy", max_depth=15)
    t7 = t7.fit(x_train, y_train)
    labels_predicted = t7.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T7:", accuracy)
    tree_to_accuracy[t7] = accuracy

    # max depth = 15, split criteria = gini
    t8 = DecisionTreeClassifier(criterion="gini", max_depth=15)
    t8 = t8.fit(x_train, y_train)
    labels_predicted = t8.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T8:", accuracy)
    tree_to_accuracy[t8] = accuracy

    # max depth = 20, split criteria = entropy
    t9 = DecisionTreeClassifier(criterion="entropy", max_depth=20)
    t9 = t9.fit(x_train, y_train)
    labels_predicted = t9.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T9:", accuracy)
    tree_to_accuracy[t9] = accuracy

    # max depth = 20, split criteria = gini
    t10 = DecisionTreeClassifier(criterion="gini", max_depth=20)
    t10 = t10.fit(x_train, y_train)
    labels_predicted = t10.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T10:", accuracy)
    tree_to_accuracy[t10] = accuracy

    best_tree = max(tree_to_accuracy, key=tree_to_accuracy.get)

    # visualize the best tree
    dot_data = export_graphviz(
        decision_tree=best_tree, 
        max_depth=2, 
        feature_names=vectorizer.get_feature_names(), 
        class_names=y_train, 
        filled=True
    )

    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree_graphivz")

    return best_tree, vectorizer

def compute_information_gain(id: int, tree: DecisionTreeClassifier, vectorizer: CountVectorizer):
    root_entropy = tree.tree_.impurity[id]
    left = tree.tree_.children_left[id]
    right = tree.tree_.children_right[id]
    left_entropy = tree.tree_.impurity[left]
    right_entropy = tree.tree_.impurity[right]

    left_split = tree.tree_.n_node_samples[left] / tree.tree_.n_node_samples[id]
    right_split = tree.tree_.n_node_samples[right] / tree.tree_.n_node_samples[id]

    info_gain = root_entropy - (left_split * left_entropy + right_split * right_entropy)

    features_to_encoding = vectorizer.vocabulary_
    encoding_to_features = {v: k for k, v in features_to_encoding.items()}

    print("The information gain of feature: '{}' is {}".format(encoding_to_features[tree.tree_.feature[id]], info_gain))

    return info_gain, left, right

if __name__ == "__main__":
    tree, vectorizer = select_data()
    info_gain, left, right = compute_information_gain(id=0, tree=tree, vectorizer=vectorizer)
    info_gain, left2, right2 = compute_information_gain(id=left, tree=tree, vectorizer=vectorizer)
    info_gain, left3, right3 = compute_information_gain(id=right, tree=tree, vectorizer=vectorizer)