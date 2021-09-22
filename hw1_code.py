from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

CLASS_LABELS = ["real", "fake"]

def load_data():
    # parse the fake data first
    file = open("clean_fake.txt", "r")
    fake = [headline.strip() for headline in file]
    file.close()

    # next parse the real data
    file = open("clean_real.txt", "r")
    real = [headline.strip() for headline in file]
    file.close()

    real_train, real_leftover = train_test_split(real, train_size=0.7)
    real_test, real_val = train_test_split(real_leftover, train_size=0.5)
    fake_train, fake_leftover = train_test_split(fake, train_size=0.7)
    fake_test, fake_val = train_test_split(fake_leftover, train_size=0.5)

    vectorizer = CountVectorizer()
    real_train = vectorizer.fit_transform(real_train)
    real_test = vectorizer.transform(real_test)
    real_val = vectorizer.transform(real_val)

    vectorizer1 = CountVectorizer()
    fake_train = vectorizer1.fit_transform(fake_train)
    fake_test = vectorizer1.transform(fake_test)
    fake_val = vectorizer1.transform(fake_val)


    return real_train, real_test, real_val, fake_train, fake_test, fake_val

    


if __name__ == "__main__":
    load_data()