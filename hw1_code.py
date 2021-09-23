from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import graphviz

REAL = "real"
FAKE = "fake"

def load_data():
    headlines = []
    labels = []
    # append all headlines into X, and then append its appropriate label to Y
    file = open("clean_real.txt", "r")
    headlines.extend([line.strip() for line in file])
    labels.extend([REAL] * len(headlines))
    file.close()

    prev_num_headlines = len(headlines)

    file = open("clean_fake.txt", "r")
    headlines.extend([line.strip() for line in file])
    labels.extend([FAKE] * (len(headlines) - prev_num_headlines))
    file.close()

    # now we want to split the data into training, test, and validation
    h_train, h_test, y_train, y_test = train_test_split(headlines, labels, train_size=0.7)
    h_test, h_validation, y_test, y_validation = train_test_split(h_test, y_test, train_size=0.5)

    # now vectorize the data
    vectorizer = TfidfVectorizer()
    h_train_vectorized = vectorizer.fit_transform(h_train)
    feature_names = vectorizer.get_feature_names()
    h_test_vectorized = vectorizer.transform(h_test)
    h_val_vectorized = vectorizer.transform(h_validation)

    return h_train_vectorized, h_test_vectorized, h_val_vectorized, y_train, y_test, y_validation, feature_names

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


def select_data():
    h_train, h_test, h_val, y_train, y_test, y_val, feature_names = load_data()

    tree_to_accuracy = {}   # maps decision trees to their accuracy scores

    # max_depth = 3, split_criteria = information gain
    t1 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    t1 = t1.fit(h_train, y_train)
    labels_predicted = t1.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T1:", )
    tree_to_accuracy[t1] = accuracy

    # max_depth = 3, split criteria = gini
    t2 = DecisionTreeClassifier(criterion="gini", max_depth=3)
    t2 = t2.fit(h_train, y_train)
    labels_predicted = t2.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T2:", accuracy)
    tree_to_accuracy[t2] = accuracy

    # max depth = 5, split criteria = entropy
    t3 = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t3 = t3.fit(h_train, y_train)
    labels_predicted = t3.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T3:", accuracy)
    tree_to_accuracy[t3] = accuracy

    # max depth = 5, split criteria = gini
    t4 = DecisionTreeClassifier(criterion="gini", max_depth=5)
    t4 = t4.fit(h_train, y_train)
    labels_predicted = t4.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T4:", accuracy)
    tree_to_accuracy[t4] = accuracy

    # max depth = 10, split criteria = entropy
    t5 = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    t5 = t5.fit(h_train, y_train)
    labels_predicted = t5.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T5:", accuracy)
    tree_to_accuracy[t5] = accuracy

    # max depth = 10, split criteria = gini
    t6 = DecisionTreeClassifier(criterion="gini", max_depth=10)
    t6 = t6.fit(h_train, y_train)
    labels_predicted = t6.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T6:", accuracy)
    tree_to_accuracy[t6] = accuracy

    # max depth = 15, split criteria = entropy
    t7 = DecisionTreeClassifier(criterion="entropy", max_depth=15)
    t7 = t7.fit(h_train, y_train)
    labels_predicted = t7.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T7:", accuracy)
    tree_to_accuracy[t7] = accuracy

    # max depth = 15, split criteria = gini
    t8 = DecisionTreeClassifier(criterion="gini", max_depth=15)
    t8 = t8.fit(h_train, y_train)
    labels_predicted = t8.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T8:", accuracy)
    tree_to_accuracy[t8] = accuracy

    # max depth = 20, split criteria = entropy
    t9 = DecisionTreeClassifier(criterion="entropy", max_depth=20)
    t9 = t9.fit(h_train, y_train)
    labels_predicted = t9.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T9:", accuracy)
    tree_to_accuracy[t9] = accuracy

    # max depth = 20, split criteria = gini
    t10 = DecisionTreeClassifier(criterion="gini", max_depth=20)
    t10 = t10.fit(h_train, y_train)
    labels_predicted = t10.predict(h_val)
    accuracy = accuracy_calculator(y_val, labels_predicted)
    print("Accuracy of T10:", accuracy)
    tree_to_accuracy[t10] = accuracy

    best_tree = max(tree_to_accuracy, key=tree_to_accuracy.get)

    # visualize the best tree
    dot_data = export_graphviz(
        decision_tree=best_tree, 
        max_depth=2, 
        feature_names=feature_names, 
        class_names=y_train, 
        filled=True
    )

    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree_graphivz")

    


if __name__ == "__main__":
    select_data()