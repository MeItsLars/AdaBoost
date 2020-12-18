package ru.datamining.adaboost.adaboost;

import lombok.RequiredArgsConstructor;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

@RequiredArgsConstructor
public class AdaBoost {

    private final int regressorTreeCount;
    private final int regressorTreeDepth;
    private final int regressorTreeSplitObservations;

    private final Map<Predictor, Double> trainResult = new HashMap<>();

    public void train(List<List<Double>> currentDataset, List<Double> currentResults) {
        if (currentDataset.isEmpty()) {
            throw new IllegalArgumentException("The input dataset can not be empty.");
        }

        if (currentDataset.size() != currentResults.size()) {
            throw new IllegalArgumentException("The input dataset must have the same length as the result set.");
        }

        if (regressorTreeCount <= 0) {
            throw new IllegalArgumentException("At least one regressor is required.");
        }

        // We initialize sample weights
        double[] weights = new double[currentDataset.get(0).size()];
        Arrays.fill(weights, 1.0 / (double) weights.length);

        // We start training 'regressorCount' predictors:
        for (int t = 0; t < regressorTreeCount; t++) {
            System.out.println("=========================");
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
            System.out.println(errorValues);
            System.out.println(maxErrorValue);

            // Calculate the average loss
            List<Double> losses = errorValues.stream().map(i -> i / maxErrorValue).collect(Collectors.toList());
            double averageLoss = losses.stream().mapToDouble(i -> i).average().orElseThrow();

            if (averageLoss >= 0.5) {
                System.out.println("====== ADABOOST TERMINATED EARLY =====");
                break;
            }

            // Calculate the measure of confidence in the predictor. Low beta = High confidence
            double beta = averageLoss / (1 - averageLoss);
            trainResult.put(regressionTree, beta);
            System.out.println("Saved a regression tree with beta " + beta);

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

    public double predict(List<Double> entry) {
        Map<Double, Double> predictions = new TreeMap<>();
        for (Map.Entry<Predictor, Double> mapEntry : trainResult.entrySet()) {
            predictions.put(mapEntry.getKey().predict(entry), mapEntry.getValue());
        }

        System.out.println(predictions);

        List<Double> valuePredictions = new ArrayList<>();
        List<Double> valueBetas = new ArrayList<>();
        predictions.forEach((d1, d2) -> {
            valuePredictions.add(d1);
            valueBetas.add(d2);
        });

        double totalHalfSum = 0;
        for (double beta : valueBetas) {
            totalHalfSum += Math.log(1.0 / beta);
        }
        totalHalfSum *= 0.5;

        int resultIndex = -1;
        for (int i = 0; i < valuePredictions.size(); i++) {
            double totalSum = 0;
            for (int j = 0; j <= i; j++) {
                totalSum += Math.log(1.0 / valueBetas.get(j));
            }
            if (totalSum >= totalHalfSum) {
                resultIndex = i;
            }
        }

        if (resultIndex == -1) {
            throw new IllegalStateException("The AdaBoost algorithm failed to produce a result.");
        }

        return valuePredictions.get(resultIndex);
    }
}