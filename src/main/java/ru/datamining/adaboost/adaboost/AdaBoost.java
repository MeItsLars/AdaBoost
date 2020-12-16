package ru.datamining.adaboost.adaboost;

import java.util.*;

public class AdaBoost {

    private Map<Predictor, Double> trainedClassifiers = new HashMap<>();

    public void train(List<List<String>> dataset, List<Double> expectedResults) {
        List<Predictor> predictors = new ArrayList<>();
        // TODO: Create stumps
        train(dataset, expectedResults, predictors);
    }

    public void train(List<List<String>> dataset, List<Double> expectedResults, List<Predictor> predictors) {
        // Initialize sample weights
        double[] sampleWeights = new double[dataset.size()];
        Arrays.fill(sampleWeights, 1.0 / ((double) dataset.size()));

        for (int i = 0; i < predictors.size(); i++) {
            // Determine best classifier for current weights using Weighted Gini

            // Determine amount of say for that classifier

            // Update data sample weights

            // Normalize data sample weights
        }
    }

    public double predict(List<String> entry) {
        return null;
    }

}
