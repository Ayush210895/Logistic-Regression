import unittest

from logistic_regression_from_scratch import LogisticRegressionGD, accuracy_score, generate_classification_data, train_test_split
from logistic_regression_from_scratch.model import sigmoid


class LogisticRegressionTests(unittest.TestCase):
    def test_sigmoid_is_stable(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertGreater(sigmoid(100), 0.99)
        self.assertLess(sigmoid(-100), 0.01)

    def test_model_learns_synthetic_boundary(self):
        dataset = generate_classification_data(n_samples=600, seed=3)
        train, test = train_test_split(dataset, test_size=0.3, seed=3)
        model = LogisticRegressionGD(learning_rate=0.2, max_iter=2000, l2=0.001).fit(train.features, train.labels)
        predictions = model.predict(test.features)

        self.assertGreater(accuracy_score(test.labels, predictions), 0.8)
        self.assertEqual(len(model.weights_), 3)


if __name__ == "__main__":
    unittest.main()
