from EEGDataFrame import EEGDataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

class ClassificationEEG:

    @classmethod
    def train_model_with_grid_search(self, df: EEGDataFrame, discrimination_type: str):
        x_train = df.train_df.drop(['type'], axis=1)
        y_train = df.train_df['type']
        MODELS = ['linear', 'quadratic', 'knn', 'random_forest']

        if MODELS.index(discrimination_type) == 0:
            model = LinearDiscriminantAnalysis()
            params = {'solver': ['svd', 'lsqr', 'eigen']}
        elif MODELS.index(discrimination_type) == 1:
            model = QuadraticDiscriminantAnalysis()
            params = {'store_covariance': [True, False]}
        elif MODELS.index(discrimination_type) == 2:
            model = KNeighborsClassifier()
            params = {'n_neighbors': [3, 5, 7, 9]}
        elif MODELS.index(discrimination_type) == 3:
            model = RandomForestClassifier()
            params = {'n_estimators': [50, 100, 150, 200]}

        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', verbose=3)
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        return best_model, best_params

    @classmethod
    def classification_model_train(self, df: EEGDataFrame, discrimination_type: str):

        x_train = df.train_df.drop(['type'], axis=1)
        y_train = df.train_df['type']
        MODELS = ['linear', 'quadratic', 'knn', 'random_forest']

        if MODELS.index(discrimination_type) ==  0 :
            lda = LinearDiscriminantAnalysis()
            lda.fit(x_train, y_train)
            return lda
        elif MODELS.index(discrimination_type) == 1:
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(x_train, y_train)
            return qda
        elif MODELS.index(discrimination_type) == 2:
            knn = KNeighborsClassifier(5)
            knn.fit(x_train, y_train)
            return knn
        elif MODELS.index(discrimination_type) == 3:
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(x_train, y_train)
            return rf
        
    @classmethod
    def classification_model_test(self, df: EEGDataFrame, model: QuadraticDiscriminantAnalysis):
        x_test = df.test_df.drop(['type'], axis=1)
        y_test = df.test_df['type']

        y_predicted = model.predict(x_test)
        return confusion_matrix(y_test, y_predicted), accuracy_score(y_test, y_predicted)