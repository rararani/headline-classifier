from logging import root
from types import CodeType
from typing import Dict, List
from scipy.sparse import base
from scipy.sparse.construct import rand
from scipy.sparse.csr import csr_matrix
from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import graphviz
import numpy as np

REAL = "real"
FAKE = "fake"


def read_file(file_name: str) -> List[str]:
    file = open(file_name, "r")
    data = [line.strip() for line in file]
    return data


def process_array(arr: List[str]) -> List[str]:
    new_arr = []
    for i in arr:
        new_arr.extend(i.split())
    return new_arr.copy()


def load_data(real_data: str, fake_data: str):
    real = read_file(real_data)
    labels = [REAL] * len(real)
    fake = read_file(fake_data)
    labels.extend([FAKE] * len(fake))
    data = real + fake

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, train_size=0.7, random_state=0)
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=0)

    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_val = vectorizer.transform(x_val)

    return x_train, x_val, y_train, y_val, vectorizer


def select_data(x_train: csr_matrix, x_val: csr_matrix, y_train: List[str], y_val: List[str]) -> DecisionTreeClassifier:
    tree_to_accuracy = {}   # maps decision trees to their accuracy scores

    # max_depth = 3, split_criteria = information gain
    t1 = DecisionTreeClassifier(
        criterion="entropy", max_depth=3, random_state=None)
    t1 = t1.fit(x_train, y_train)
    labels_predicted = t1.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T1:", accuracy)
    tree_to_accuracy[(t1, 1, t1.criterion, t1.max_depth)] = accuracy

    # max_depth = 3, split criteria = gini
    t2 = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=None)
    t2 = t2.fit(x_train, y_train)
    labels_predicted = t2.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T2:", accuracy)
    tree_to_accuracy[(t2, 2, t2.criterion, t2.max_depth)] = accuracy

    # max depth = 5, split criteria = entropy
    t3 = DecisionTreeClassifier(
        criterion="entropy", max_depth=5, random_state=None)
    t3 = t3.fit(x_train, y_train)
    labels_predicted = t3.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T3:", accuracy)
    tree_to_accuracy[(t3, 3, t3.criterion, t3.max_depth)] = accuracy

    # max depth = 5, split criteria = gini
    t4 = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=None)
    t4 = t4.fit(x_train, y_train)
    labels_predicted = t4.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T4:", accuracy)
    tree_to_accuracy[(t4, 4, t4.criterion, t4.max_depth)] = accuracy

    # max depth = 10, split criteria = entropy
    t5 = DecisionTreeClassifier(
        criterion="entropy", max_depth=10, random_state=None)
    t5 = t5.fit(x_train, y_train)
    labels_predicted = t5.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T5:", accuracy)
    tree_to_accuracy[(t5, 5, t5.criterion, t5.max_depth)] = accuracy

    # max depth = 10, split criteria = gini
    t6 = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=None)
    t6 = t6.fit(x_train, y_train)
    labels_predicted = t6.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T6:", accuracy)
    tree_to_accuracy[(t6, 6, t6.criterion, t6.max_depth)] = accuracy

    # max depth = 15, split criteria = entropy
    t7 = DecisionTreeClassifier(
        criterion="entropy", max_depth=15, random_state=None)
    t7 = t7.fit(x_train, y_train)
    labels_predicted = t7.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T7:", accuracy)
    tree_to_accuracy[(t7, 7, t7.criterion, t7.max_depth)] = accuracy

    # max depth = 15, split criteria = gini
    t8 = DecisionTreeClassifier(criterion="gini", max_depth=15, random_state=None)
    t8 = t8.fit(x_train, y_train)
    labels_predicted = t8.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T8:", accuracy)
    tree_to_accuracy[(t8, 8, t8.criterion, t8.max_depth)] = accuracy

    # max depth = 20, split criteria = entropy
    t9 = DecisionTreeClassifier(
        criterion="entropy", max_depth=20, random_state=None)
    t9 = t9.fit(x_train, y_train)
    labels_predicted = t9.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T9:", accuracy)
    tree_to_accuracy[(t9, 9, t9.criterion, t9.max_depth)] = accuracy

    # max depth = 20, split criteria = gini
    t10 = DecisionTreeClassifier(
        criterion="gini", max_depth=20, random_state=None)
    t10 = t10.fit(x_train, y_train)
    labels_predicted = t10.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T10:", accuracy)
    tree_to_accuracy[(t10, 10, t10.criterion, t10.max_depth)] = accuracy

    best_tree = max(tree_to_accuracy, key=tree_to_accuracy.get)

    print("\nThe most accurate tree is Tree #{} with hyperparamters: split_criteria = {} and max_depth = {}".format(
        best_tree[1], best_tree[2], best_tree[3]
        )
    )

    return best_tree[0]


def compute_information_gain(keyword: int, clf: DecisionTreeClassifier, keyword_to_encoding: Dict[str, int]):
    keyword_code = keyword_to_encoding[keyword]

    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature

    stack = [0]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        i = stack.pop()
        code = feature[i]
        # print("Code: {} Key_code: {} Eq: {}".format(code, keyword_code, code == keyword_code))
        if code == keyword_code:
            left_id = children_left[i]
            right_id = children_right[i]
            node_id = i
            break

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[i] != children_right[i]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append(children_left[i])
            stack.append(children_right[i])

    if tree.criterion == "gini":
        if left_id < 0 or right_id < 0:
            return 2 * tree.tree_.impurity[node_id]

        root_entropy = 2 * tree.tree_.impurity[node_id]
        left_entropy = 2 * tree.tree_.impurity[left_id]
        right_entropy = 2 * tree.tree_.impurity[right_id]
        prob_left = tree.tree_.n_node_samples[left_id] / tree.tree_.n_node_samples[node_id]
        prob_right = tree.tree_.n_node_samples[right_id] / tree.tree_.n_node_samples[node_id]

        return root_entropy - (prob_left * left_entropy + prob_right * right_entropy)    
    else:
        return tree.tree_.impurity[node_id]


        


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

if __name__ == "__main__":
    x_train, x_val, y_train, y_val, vectorizer = load_data("clean_real.txt", "clean_fake.txt")
    tree = select_data(x_train, x_val, y_train, y_val)
    
    dot_data = export_graphviz(
        decision_tree=tree, 
        max_depth=2, 
        feature_names=vectorizer.get_feature_names(), 
        class_names=y_train, 
        filled=True
    )

    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree_graphivz")

    keyword_to_encoding = vectorizer.vocabulary_
    encoding_to_keyword = {value: key for (key, value) in keyword_to_encoding.items()}
    
    info_gain1 = compute_information_gain("hillary", tree, keyword_to_encoding)
    print(info_gain1)