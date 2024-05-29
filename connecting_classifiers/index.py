from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_classification_models(n_samples, noise, random_state, test_size, n_estimators, max_iter):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    tree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=random_state)
    tree_gini = DecisionTreeClassifier(random_state=random_state)
    tree_entropy.fit(X_train, y_train)
    tree_gini.fit(X_train, y_train)

    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    forest.fit(X_train, y_train)

    logistic_regression = LogisticRegression(max_iter=max_iter, random_state=random_state)
    svm = SVC(random_state=random_state, probability=True)
    logistic_regression.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    print("Dokładność drzewa decyzyjnego (entropia):", accuracy_score(y_test, tree_entropy.predict(X_test)))
    print("Dokładność drzewa decyzyjnego (Gini):", accuracy_score(y_test, tree_gini.predict(X_test)))
    print("Dokładność lasu losowego:", accuracy_score(y_test, forest.predict(X_test)))
    print("Dokładność regresji logistycznej:", accuracy_score(y_test, logistic_regression.predict(X_test)))
    print("Dokładność SVM:", accuracy_score(y_test, svm.predict(X_test)))

    voting_clf_1 = VotingClassifier(estimators=[('lr', logistic_regression), ('svm', svm), ('rf', forest)])
    voting_clf_1.fit(X_train, y_train)
    print("Dokładność połączonego klasyfikatora (hard):", accuracy_score(y_test, voting_clf_1.predict(X_test)))

    voting_clf_2 = VotingClassifier(estimators=[('lr', logistic_regression), ('svm', svm), ('rf', forest)],
                                    voting='soft')
    voting_clf_2.fit(X_train, y_train)
    print("Dokładność połączonego klasyfikatora (soft):", accuracy_score(y_test, voting_clf_2.predict(X_test)))

    classifiers = [tree_gini, forest, logistic_regression, svm, voting_clf_2]
    classifier_names = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'Voting Classifier']

    plt.figure(figsize=(10, 8))
    for clf, name in zip(classifiers, classifier_names):
        y_pred = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC_Curves.png')
    plt.show()
