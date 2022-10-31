from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base interface for machine learning models.
    """

    @abstractmethod
    def fit(self, x, y):
        """
        Train the model on the given data.
        :param x: The input data.
        :param y: The output data.
        :return: The trained model.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Predict the labels for the given data.
        :param x: The input data.
        :return: The predicted labels.
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Serialize the model to the given path.
        :param path: The path to save the model to.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load the model from the given path.
        :param path: The path to load the model from.
        """
        pass
