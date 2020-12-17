package ru.datamining.adaboost.adaboost;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class AdaBoost {

    private Map<Predictor, Double> trainedClassifiers = new HashMap<>();

    public void train(List<List<Double>> dataset, List<Double> expectedResults) {
        List<Predictor> predictors = new ArrayList<>();

        for (int i = 0; i < dataset.get(0).size(); i++) {
            predictors.add(new RegressionStump(i));
        }

        train(dataset, expectedResults, predictors);
    }

    public void train(List<List<Double>> dataset, List<Double> expectedResults, List<Predictor> predictors) {
        // Initialize sample weights
        double[] sampleWeights = new double[dataset.size()];
        Arrays.fill(sampleWeights, 1.0 / ((double) dataset.size()));

        for (int i = 0; i < predictors.size(); i++) {
            Predictor predictor = predictors.get(i);

            // Train the predictor
            predictor.train(dataset, expectedResults);
            List<Double> yPredict = dataset.stream().map(predictor::predict).collect(Collectors.toList());
            List<Double> errors = IntStream.range(0, yPredict.size()).mapToDouble(v -> yPredict.get(v) - expectedResults.get(v)).boxed().collect(Collectors.toList());

            Optional<Double> opt = errors.stream().mapToDouble(d -> d).boxed().max(Double::compareTo);
            if (opt.isPresent() && opt.get() != 0) {
                errors = errors.stream().map(v -> Math.pow(v / opt.get(), 2)).collect(Collectors.toList());
            }
            List<Double> finalErrors = errors;
            double trainingError = IntStream.range(0, errors.size()).mapToDouble(v -> finalErrors.get(v) * sampleWeights[v]).sum();

            // Determine amount of say for that classifier
            double alpha = 0.5 * Math.log1p((1 - trainingError) / trainingError);
            trainedClassifiers.put(predictor, alpha);

            // Update data sample weights
            double beta = trainingError / (1 - trainingError);
            if (i != predictors.size() - 1) {
                for (int w = 0; w < sampleWeights.length; w++) {
                    sampleWeights[w] *= Math.pow(beta, w - finalErrors.get(w));
                }
            }

            // Normalize data sample weights
            double sum = Arrays.stream(sampleWeights).sum();
            for (int w = 0; w < sampleWeights.length; w++) {
                sampleWeights[w] /= sum;
            }
        }
    }

    public double predict(List<Double> entry) {
        double result = 0;
        double totalAmountOfSay = 0;
        for (Map.Entry<Predictor, Double> classifier : trainedClassifiers.entrySet()) {
            result += classifier.getValue() * classifier.getKey().predict(entry);
            totalAmountOfSay += classifier.getValue();
        }
        return result / totalAmountOfSay;
    }

}
