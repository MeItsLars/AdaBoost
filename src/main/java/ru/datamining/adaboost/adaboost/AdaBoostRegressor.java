package ru.datamining.adaboost.adaboost;

import lombok.RequiredArgsConstructor;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Class that represents an AdaBoost regressor.
 */
@RequiredArgsConstructor
public class AdaBoostRegressor {

    // The amount of regression trees that this AdaBoost instance will use
    private final int regressorTreeCount;
    // The maximum depth of the regression trees that this AdaBoost instance will use
    private final int regressorTreeDepth;
    // The minimum number of observations in a regression tree before it can split
    private final int regressorTreeSplitObservations;

    // After training: contains the resulting regressors and their importance
    private final Map<Regressor, Double> trainResult = new HashMap<>();

    /**
     * Trains the AdaBoost regressor on the given input data, with the given results
     * The size of the 'data' list should be equal to the size of the 'expectedResults' list
     *
     * @param currentDataset The input data, formatted into a list of data entries
     * @param currentResults The expected results, formatted into a list of values
     */
    public void train(List<List<Double>> currentDataset, List<Double> currentResults) {
        // Check that the dataset is not empty
        if (currentDataset.isEmpty()) {
            throw new IllegalArgumentException("The input dataset can not be empty.");
        }

        // Check that the size of the dataset and the results is the same
        if (currentDataset.size() != currentResults.size()) {
            throw new IllegalArgumentException("The input dataset must have the same length as the result set.");
        }

        // Check that there are is at least 1 regression tree
        if (regressorTreeCount <= 0) {
            throw new IllegalArgumentException("At least one regressor is required.");
        }

        // We initialize sample weights
        double[] weights = new double[currentDataset.size()];
        Arrays.fill(weights, 1.0 / (double) weights.length);

        // We start training 'regressorTreeCount' predictors:
        for (int t = 0; t < regressorTreeCount; t++) {
            // We randomly re-create the training set
            List<List<Double>> dataset = new ArrayList<>();
            List<Double> results = new ArrayList<>();
            for (int i = 0; i < weights.length; i++) {
                double random = ThreadLocalRandom.current().nextDouble(1.0);
                int index = 0;
                while (random > weights[index]) {
                    random -= weights[index];
                    index++;
                }
                dataset.add(currentDataset.get(index));
                results.add(currentResults.get(index));
            }
            currentDataset = dataset;
            currentResults = results;
            Arrays.fill(weights, 1.0 / (double) weights.length);

            // We create and train a new regression tree
            RegressionTree regressionTree = new RegressionTree(regressorTreeDepth, regressorTreeSplitObservations);
            regressionTree.train(dataset, results);

            // We calculate the training error of the algorithm
            List<Double> errorValues = IntStream.range(0, weights.length)
                    .mapToDouble(i -> Math.abs(regressionTree.predict(dataset.get(i)) - results.get(i)))
                    .boxed()
                    .collect(Collectors.toList());
            double maxErrorValue = errorValues.stream().mapToDouble(i -> i).max().orElseThrow() + 0.000001;

            // Calculate the average loss
            List<Double> losses = errorValues.stream().map(i -> i / maxErrorValue).collect(Collectors.toList());
            double averageLoss = losses.stream().mapToDouble(i -> i).average().orElseThrow();

            if (averageLoss >= 0.5) {
                break;
            }

            // Calculate the measure of confidence in the predictor. Low beta = High confidence
            double beta = averageLoss / (1 - averageLoss);
            trainResult.put(regressionTree, beta);

            // Update all weights
            for (int i = 0; i < weights.length; i++) {
                weights[i] = weights[i] * Math.pow(beta, 1 - losses.get(i));
            }

            // Normalize all weights:
            double weightsSum = Arrays.stream(weights).sum();
            for (int i = 0; i < weights.length; i++) {
                weights[i] = weights[i] / weightsSum;
            }
        }
    }

    /**
     * Predicts the output value of the given input entry using AdaBoost
     *
     * @param entry The input entry
     * @return The predicted output
     */
    public double predict(List<Double> entry) {
        double totalSum = trainResult.values().stream().mapToDouble(d -> d).sum();
        return trainResult.entrySet().stream().mapToDouble(e -> (e.getKey().predict(entry) * e.getValue()) / totalSum).sum();
    }
}