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

    real_vectorizer = CountVectorizer()
    real = real_vectorizer.fit_transform(real)
    fake_vectorizer = CountVectorizer()
    fake = fake_vectorizer.fit_transform(fake)

    # next split the real and fake data sets into training, testing, and validation sets
    real_train, real_leftover = train_test_split(real, train_size=0.7)
    real_test, real_val = train_test_split(real_leftover, train_size=0.5)
    fake_train, fake_leftoever = train_test_split(fake, train_size=0.7)
    fake_test, fake_val = train_test_split(fake_leftoever, train_size=0.5)

    return real_train, real_test, real_val, fake_train, fake_test, fake_val

    


if __name__ == "__main__":
    load_data()