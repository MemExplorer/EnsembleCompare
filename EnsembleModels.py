import pickle
import numpy as np
from sklearn.model_selection import (
    KFold,
    train_test_split,
    cross_val_score,
    cross_val_predict,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score


class EnsembleModel:
    def __init__(self, model_instance):
        self.__model_instance = model_instance

    def fit(self, x_data, y_data):
        self.__model_instance.fit(x_data, y_data)

    def predict(self, input):
        return self.__model_instance.predict(input)

    def predict_a(self, input):
        result = np.squeeze(self.__model_instance.predict_proba(input))
        max_index = np.argmax(result)
        return (
            round(result[max_index] * 100, 2),
            self.__model_instance.classes_[max_index],
        )

    def benchmark_split(self, x_data, y_data, test_size=0.3):
        """Gets the benchmark of the model yung the split method

        Returns
        -------
        A tuple that contains the following in order: confusion matrix, accuracy score, and kappa score
        """
        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_size
        )
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        ac_score = accuracy_score(y_test, y_pred)
        kappa_score = cohen_kappa_score(y_test, y_pred)
        return conf_matrix, ac_score, kappa_score

    def benchmark_kfold(self, kfold_instance, x_data, y_data):
        """Gets the benchmark of the model yung the kfold method

        Returns
        -------
        A tuple that contains the following in order: confusion matrix, accuracy score, and kappa score
        """
        accuracy_score = cross_val_score(
            self.__model_instance, x_data, y_data, cv=kfold_instance
        )
        y_pred = cross_val_predict(
            self.__model_instance, x_data, y_data, cv=kfold_instance
        )
        cf = confusion_matrix(y_data, y_pred)
        kappa_score = cohen_kappa_score(y_data, y_pred)
        return cf, accuracy_score.mean(), kappa_score

    def export_model(self, model_path="model.pickle"):
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    # static function
    def load_from_model(model_path="model.pickle"):
        with open(model_path, "rb") as f:
            return pickle.load(f)


class AdaBoosting(EnsembleModel):
    def __init__(self, splits=5, num_trees=150, seed=None):
        self.__kfold = KFold(
            n_splits=splits, random_state=seed, shuffle=seed is not None
        )
        model = AdaBoostClassifier(
            n_estimators=num_trees, random_state=seed, algorithm="SAMME"
        )
        super().__init__(model)

    def benchmark_split(self, x_data, y_data, test_size=0.3):
        return super().benchmark_split(x_data, y_data, test_size)

    def benchmark_kfold(self, x_data, y_data):
        return super().benchmark_kfold(self.__kfold, x_data, y_data)


class BaggingRandomForest(EnsembleModel):
    def __init__(self, splits=5, num_trees=150, seed=None):
        self.__kfold = KFold(
            n_splits=splits, random_state=seed, shuffle=seed is not None
        )
        model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
        super().__init__(model)

    def benchmark_split(self, x_data, y_data, test_size=0.3):
        return super().benchmark_split(x_data, y_data, test_size)

    def benchmark_kfold(self, x_data, y_data):
        return super().benchmark_kfold(self.__kfold, x_data, y_data)


class BaggingClassicDecisionTree(EnsembleModel):
    def __init__(self, splits=5, num_trees=150, seed=None):
        self.__kfold = KFold(
            n_splits=splits, random_state=seed, shuffle=seed is not None
        )
        cart = DecisionTreeClassifier(random_state=seed)
        model = BaggingClassifier(
            estimator=cart, n_estimators=num_trees, random_state=seed
        )
        super().__init__(model)

    def benchmark_split(self, x_data, y_data, test_size=0.3):
        return super().benchmark_split(x_data, y_data, test_size)

    def benchmark_kfold(self, x_data, y_data):
        return super().benchmark_kfold(self.__kfold, x_data, y_data)
