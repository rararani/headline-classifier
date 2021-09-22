from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
    h_test_vectorized = vectorizer.transform(h_test)
    h_val_vectorized = vectorizer.transform(h_validation)

    return h_train_vectorized, h_test_vectorized, h_val_vectorized, y_train, y_test, y_validation

def select_data():
    h_train, h_test, h_val, y_train, y_test, y_val = load_data()
    
    # max_depth = 3, split_criteria = information gain
    t1 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    t1 = t1.fit(h_train, y_train)
    labels_predicted = t1.predict(h_val)
    print("Accuracy of T1:", metrics.accuracy_score(y_val, labels_predicted))

    # max_depth = 3, split criteria = gini
    t2 = DecisionTreeClassifier(criterion="gini", max_depth=3)
    t2 = t2.fit(h_train, y_train)
    labels_predicted = t2.predict(h_val)
    print("Accuracy of T2:", metrics.accuracy_score(y_val, labels_predicted))

    # max depth = 5, split criteria = entropy
    t3 = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t3 = t3.fit(h_train, y_train)
    labels_predicted = t3.predict(h_val)
    print("Accuracy of T3:", metrics.accuracy_score(y_val, labels_predicted))

    # max depth = 5, split criteria = gini
    t4 = DecisionTreeClassifier(criterion="gini", max_depth=5)
    t4 = t4.fit(h_train, y_train)
    labels_predicted = t4.predict(h_val)
    print("Accuracy of T4:", metrics.accuracy_score(y_val, labels_predicted))

    # max depth = 10, split criteria = entropy
    t5 = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    t5 = t5.fit(h_train, y_train)
    labels_predicted = t5.predict(h_val)
    print("Accuracy of T5:", metrics.accuracy_score(y_val, labels_predicted))

    # max depth = 10, split criteria = gini
    t6 = DecisionTreeClassifier(criterion="gini", max_depth=10)
    t6 = t6.fit(h_train, y_train)
    labels_predicted = t6.predict(h_val)
    print("Accuracy of T6:", metrics.accuracy_score(y_val, labels_predicted))

    # max depth = 15, split criteria = entropy
    t7 = DecisionTreeClassifier(criterion="entropy", max_depth=15)
    t7 = t7.fit(h_train, y_train)
    labels_predicted = t7.predict(h_val)
    print("Accuracy of T7:", metrics.accuracy_score(y_val, labels_predicted))

    # max depth = 15, split criteria = gini
    t8 = DecisionTreeClassifier(criterion="gini", max_depth=15)
    t8 = t8.fit(h_train, y_train)
    labels_predicted = t8.predict(h_val)
    print("Accuracy of T8:", metrics.accuracy_score(y_val, labels_predicted))

    # max depth = 20, split criteria = entropy
    t9 = DecisionTreeClassifier(criterion="entropy", max_depth=20)
    t9 = t9.fit(h_train, y_train)
    labels_predicted = t9.predict(h_val)
    print("Accuracy of T9:", metrics.accuracy_score(y_val, labels_predicted))

    # max depth = 20, split criteria = gini
    t10 = DecisionTreeClassifier(criterion="gini", max_depth=20)
    t10 = t10.fit(h_train, y_train)
    labels_predicted = t10.predict(h_val)
    print("Accuracy of T10:", metrics.accuracy_score(y_val, labels_predicted))


    


if __name__ == "__main__":
    select_data()