package ru.datamining.adaboost.adaboost;

import java.util.*;

public class AdaBoost {

    private Map<Classifier, Double> trainedClassifiers = new HashMap<>();

    public void train(List<List<String>> dataset, List<Double> expectedResults) {
        List<Classifier> classifiers = new ArrayList<>();
        // TODO: Create stumps
        train(dataset, expectedResults, classifiers);
    }

    public void train(List<List<String>> dataset, List<Double> expectedResults, List<Classifier> classifiers) {
        // Initialize sample weights
        double[] sampleWeights = new double[dataset.size()];
        Arrays.fill(sampleWeights, 1.0 / ((double) dataset.size()));

        for (int i = 0; i < classifiers.size(); i++) {
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
