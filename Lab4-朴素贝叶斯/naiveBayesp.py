import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    
    df = df.dropna().drop_duplicates()
    
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', x).lower())
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    return labels, texts

def split_dataset(texts, labels, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return X_train, X_test, y_train, y_test

def create_bow_features():
    vectorizer = CountVectorizer(stop_words='english')
    return vectorizer

def train_naive_bayes(X_train, y_train):
    
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\n[4] 模型评估结果：")
    print(f"    准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常邮件(ham)', '垃圾邮件(spam)']))

if __name__ == "__main__":
    print("=" * 60)
    print("多项式朴素贝叶斯 垃圾邮件分类器")
    print("=" * 60)

    DATA_PATH = "spam.csv"
    
    print("\n[1] 读取数据...")
    labels, texts = load_and_clean_data(DATA_PATH)
    print(f"    有效样本数: {len(texts)}")
    print(f"    ham (正常邮件): {labels.count(0)}")
    print(f"    spam (垃圾邮件): {labels.count(1)}")
    
    print(f"\n[2] 划分数据集（测试集比例 = 20%）...")
    X_train, X_test, y_train, y_test = split_dataset(texts, labels)
    print(f"    训练集大小: {len(X_train)}")
    print(f"    测试集大小: {len(X_test)}")
    
    vectorizer = create_bow_features()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    
    print("\n[3] 训练多项式朴素贝叶斯分类器（Laplace平滑 α=1.0）...")
    nb_model = train_naive_bayes(X_train_bow, y_train)
    
    evaluate_model(nb_model, X_test_bow, y_test)