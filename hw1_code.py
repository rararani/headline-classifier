from ctypes import sizeof
from scipy.sparse.sputils import validateaxis
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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

    


if __name__ == "__main__":
    print(load_data())