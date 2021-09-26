
from typing import List
from scipy.sparse.csr import csr_matrix
from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import graphviz

REAL = "real"
FAKE = "fake"

def load_data(real_data, fake_data):
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
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_val_vectorized = vectorizer.transform(x_val)

    return x_train_vectorized, x_val_vectorized, x_train, y_train, y_val, vectorizer.get_feature_names()


def select_data(x_train, x_val, y_train, y_val):
    tree_to_accuracy = {}   # maps decision trees to their accuracy scores

    # max_depth = 3, split_criteria = information gain
    t1 = DecisionTreeClassifier(
        criterion="entropy", max_depth=3, random_state=0)
    t1 = t1.fit(x_train, y_train)
    labels_predicted = t1.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T1:", accuracy)
    tree_to_accuracy[(t1, 1, t1.criterion, t1.max_depth)] = accuracy

    # max_depth = 3, split criteria = gini
    t2 = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=0)
    t2 = t2.fit(x_train, y_train)
    labels_predicted = t2.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T2:", accuracy)
    tree_to_accuracy[(t2, 2, t2.criterion, t2.max_depth)] = accuracy

    # max depth = 5, split criteria = entropy
    t3 = DecisionTreeClassifier(
        criterion="entropy", max_depth=5, random_state=0)
    t3 = t3.fit(x_train, y_train)
    labels_predicted = t3.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T3:", accuracy)
    tree_to_accuracy[(t3, 3, t3.criterion, t3.max_depth)] = accuracy

    # max depth = 5, split criteria = gini
    t4 = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=0)
    t4 = t4.fit(x_train, y_train)
    labels_predicted = t4.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T4:", accuracy)
    tree_to_accuracy[(t4, 4, t4.criterion, t4.max_depth)] = accuracy

    # max depth = 10, split criteria = entropy
    t5 = DecisionTreeClassifier(
        criterion="entropy", max_depth=10, random_state=0)
    t5 = t5.fit(x_train, y_train)
    labels_predicted = t5.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T5:", accuracy)
    tree_to_accuracy[(t5, 5, t5.criterion, t5.max_depth)] = accuracy

    # max depth = 10, split criteria = gini
    t6 = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=0)
    t6 = t6.fit(x_train, y_train)
    labels_predicted = t6.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T6:", accuracy)
    tree_to_accuracy[(t6, 6, t6.criterion, t6.max_depth)] = accuracy

    # max depth = 15, split criteria = entropy
    t7 = DecisionTreeClassifier(
        criterion="entropy", max_depth=15, random_state=0)
    t7 = t7.fit(x_train, y_train)
    labels_predicted = t7.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T7:", accuracy)
    tree_to_accuracy[(t7, 7, t7.criterion, t7.max_depth)] = accuracy

    # max depth = 15, split criteria = gini
    t8 = DecisionTreeClassifier(criterion="gini", max_depth=15, random_state=0)
    t8 = t8.fit(x_train, y_train)
    labels_predicted = t8.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T8:", accuracy)
    tree_to_accuracy[(t8, 8, t8.criterion, t8.max_depth)] = accuracy

    # max depth = 20, split criteria = entropy
    t9 = DecisionTreeClassifier(
        criterion="entropy", max_depth=20, random_state=0)
    t9 = t9.fit(x_train, y_train)
    labels_predicted = t9.predict(x_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T9:", accuracy)
    tree_to_accuracy[(t9, 9, t9.criterion, t9.max_depth)] = accuracy

    # max depth = 20, split criteria = gini
    t10 = DecisionTreeClassifier(
        criterion="gini", max_depth=20, random_state=0)
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

def compute_information_gain(keyword, x_train, y_train):
    data = combine_array(x_train, y_train)
    left = []
    right = []
    for item in data:
        if keyword in item[0]:
            left.append(item)
        else:
            right.append(item)
    
    n_real = y_train.count(REAL)
    n_samples = len(x_train)
    p_real = n_real / n_samples # P(Y=real)

    root_entropy = entropy([p_real, 1-p_real], base=2)  # H(Y)

    n_real_left = 0
    for item in left:
        if item[1] == REAL:
            n_real_left += 1
    
    cp_real_left = n_real_left / len(left)
    left_entropy = entropy([cp_real_left, 1-cp_real_left], base=2)    # H(Y|left)

    n_real_right = 0
    for item in right:
        if item[1] == REAL:
            n_real_right += 1
    
    cp_real_right = n_real_right / len(right)
    right_entropy = entropy([cp_real_right, 1 - cp_real_right], base=2) # H(Y|right)

    return root_entropy - (len(left) / len(x_train) * left_entropy + len(right) / len(x_train) * right_entropy)

def read_file(file_name):
    file = open(file_name, "r")
    data = [line.strip() for line in file]
    return data

def accuracy_calculator(y_true, y_pred):
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

def combine_array(headlines, labels):
    arr = []
    for i in range(len(headlines)):
        arr.append((headlines[i], labels[i]))
    return arr.copy()

if __name__ == "__main__":
    x_train_vectorized, x_val_vectorized, x_train, y_train, y_val, features = load_data("clean_real.txt", "clean_fake.txt")
    tree = select_data(x_train_vectorized, x_val_vectorized, y_train, y_val)
    
    dot_data = export_graphviz(
        decision_tree=tree, 
        max_depth=2, 
        feature_names=features, 
        class_names=y_train, 
        filled=True
    )

    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree_graphivz")

    print("The information gain of the word: the is", compute_information_gain("the", x_train, y_train))
    print("The information gain of the word: donald is", compute_information_gain("donald", x_train, y_train))
    print("The information gain of the word: trumps is", compute_information_gain("trumps", x_train, y_train))
