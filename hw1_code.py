from typing import List
from numpy.lib.function_base import select
from scipy.sparse import base
from scipy.sparse.csr import csr_matrix
from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import graphviz

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

    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.7, random_state=0)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_val = vectorizer.transform(x_val)

    return x_train, x_val, y_train, y_val, vectorizer.get_feature_names(), process_array(real), process_array(fake)

def select_data(x_train: csr_matrix, x_val: csr_matrix, y_train: List[str], y_val: List[str]) -> DecisionTreeClassifier:
    tree_to_accuracy = {}   # maps decision trees to their accuracy scores

    # max_depth = 3, split_criteria = information gain
    t1 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    t1 = t1.fit(x_train, y_train)
    labels_predicted = t1.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T1:", accuracy)
    tree_to_accuracy[(t1, 1, t1.criterion, t1.max_depth)] = accuracy

    # max_depth = 3, split criteria = gini
    t2 = DecisionTreeClassifier(criterion="gini", max_depth=3)
    t2 = t2.fit(x_train, y_train)
    labels_predicted = t2.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T2:", accuracy)
    tree_to_accuracy[(t2, 2, t2.criterion, t2.max_depth)] = accuracy

    # max depth = 5, split criteria = entropy
    t3 = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t3 = t3.fit(x_train, y_train)
    labels_predicted = t3.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T3:", accuracy)
    tree_to_accuracy[(t3, 3, t3.criterion, t3.max_depth)] = accuracy

    # max depth = 5, split criteria = gini
    t4 = DecisionTreeClassifier(criterion="gini", max_depth=5)
    t4 = t4.fit(x_train, y_train)
    labels_predicted = t4.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T4:", accuracy)
    tree_to_accuracy[(t4, 4, t4.criterion, t4.max_depth)] = accuracy

    # max depth = 10, split criteria = entropy
    t5 = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    t5 = t5.fit(x_train, y_train)
    labels_predicted = t5.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T5:", accuracy)
    tree_to_accuracy[(t5, 5, t5.criterion, t5.max_depth)] = accuracy

    # max depth = 10, split criteria = gini
    t6 = DecisionTreeClassifier(criterion="gini", max_depth=10)
    t6 = t6.fit(x_train, y_train)
    labels_predicted = t6.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T6:", accuracy)
    tree_to_accuracy[(t6, 6, t6.criterion, t6.max_depth)] = accuracy

    # max depth = 15, split criteria = entropy
    t7 = DecisionTreeClassifier(criterion="entropy", max_depth=15)
    t7 = t7.fit(x_train, y_train)
    labels_predicted = t7.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T7:", accuracy)
    tree_to_accuracy[(t7, 7, t7.criterion, t7.max_depth)] = accuracy

    # max depth = 15, split criteria = gini
    t8 = DecisionTreeClassifier(criterion="gini", max_depth=15)
    t8 = t8.fit(x_train, y_train)
    labels_predicted = t8.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T8:", accuracy)
    tree_to_accuracy[(t8, 8, t8.criterion, t8.max_depth)] = accuracy

    # max depth = 20, split criteria = entropy
    t9 = DecisionTreeClassifier(criterion="entropy", max_depth=20)
    t9 = t9.fit(x_train, y_train)
    labels_predicted = t9.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T9:", accuracy)
    tree_to_accuracy[(t9, 9, t9.criterion, t9.max_depth)] = accuracy

    # max depth = 20, split criteria = gini
    t10 = DecisionTreeClassifier(criterion="gini", max_depth=20)
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

def compute_information_gain(keyword: str, real: List[str], fake: List[str]) -> float:
    # total = Decimal(len(real)) + Decimal(len(fake))
    total = (len(real) + len(fake))
    prob_real = len(real) / total
    root_entropy = entropy([prob_real, 1 - prob_real], base=2)

    prob_keyword = (real.count(keyword) + fake.count(keyword)) / total
    prob_real_keyword = real.count(keyword) / total
    prob_fake_keyword = fake.count(keyword) / total
    cond_prob_real = prob_real_keyword / prob_keyword
    cond_prob_fake = prob_fake_keyword / prob_keyword
    cond_entropy = entropy([cond_prob_real, cond_prob_fake], base=2)

    return root_entropy - cond_entropy

    





# def entropy_calculator(prob1: float, prob2: float) -> float:
#     return -1 * (prob1 * math.log2(prob1) + prob2 * math.log2(prob2))

# def compute_information_gain(word: str, real_train: List[str], fake_train: List[str]) -> float:
#     # first calculate P(Y = real) and P(Y = fake)
#     prob_real = len(real_train) / (len(real_train) + len(fake_train))
#     # H(Y)
#     root_entropy = entropy_calculator(prob_real, 1 - prob_real)
#     # calculate P(Y = real, X = word)
#     prob_real_word = real_train.count(word) / (len(real_train) + len(fake_train))
#     # next calculate P(X = word)
#     prob_word = (real_train.count(word) + fake_train.count(word)) / (len(real_train) + len(fake_train))
#     # P(Y = real|X = word)
#     cond_prob_real = prob_real_word / prob_word
#     # P(Y = fake, X = word)
#     prob_fake_word = fake_train.count(word) / (len(real_train) + len(fake_train))
#     # P(Y = fake|X = word)
#     cond_prob_fake = prob_fake_word / prob_word
#     # H(Y|X = word)
#     cond_entropy = entropy_calculator(cond_prob_real, cond_prob_fake)

#     return root_entropy - cond_entropy




# def compute_information_gain(id: int, tree: DecisionTreeClassifier, vectorizer: CountVectorizer):
#     root_entropy = tree.tree_.impurity[id]
#     left = tree.tree_.children_left[id]
#     right = tree.tree_.children_right[id]
#     left_entropy = tree.tree_.impurity[left]
#     right_entropy = tree.tree_.impurity[right]

#     left_split = tree.tree_.n_node_samples[left] / tree.tree_.n_node_samples[id]
#     right_split = tree.tree_.n_node_samples[right] / tree.tree_.n_node_samples[id]

#     info_gain = root_entropy - (left_split * left_entropy + right_split * right_entropy)

#     features_to_encoding = vectorizer.vocabulary_
#     encoding_to_features = {v: k for k, v in features_to_encoding.items()}

#     print("The information gain of feature: '{}' is {}".format(encoding_to_features[tree.tree_.feature[id]], info_gain))

#     return info_gain, left, right

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
    x_train, x_val, y_train, y_val, features, real, fake = load_data("clean_real.txt", "clean_fake.txt")
    tree = select_data(x_train, x_val, y_train, y_val)
    
    dot_data = export_graphviz(
        decision_tree=tree, 
        max_depth=2, 
        feature_names=features, 
        class_names=y_train, 
        filled=True
    )

    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree_graphivz")

    info_gain1 = compute_information_gain("trump", real, fake)
    info_gain2 = compute_information_gain("market", real, fake)
    info_gain3 = compute_information_gain("the", real, fake)

    print("The information gain for the word: trump is", info_gain1)
    print("The information gain for the word: market is", info_gain2)
    print("The information gain for the word: the is", info_gain3)