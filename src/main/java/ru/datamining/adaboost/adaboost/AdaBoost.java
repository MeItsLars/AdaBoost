package ru.datamining.adaboost.adaboost;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class AdaBoost {

    private Map<Predictor, Double> trainedClassifiers = new HashMap<>();

    public void train(List<List<Double>> dataset, List<Double> expectedResults, int regressorCount) {
        // Initialize sample weights
        double[] sampleWeights = new double[dataset.size()];
        Arrays.fill(sampleWeights, 1.0 / ((double) dataset.size()));

        for (int i = 0; i < regressorCount; i++) {
            System.out.println("========== RUN " + i + " ==========");
            System.out.println(dataset);
            System.out.println(expectedResults);
            Predictor predictor = new RegressionTree(3, 5);

            // Train the predictor
            predictor.train(dataset, expectedResults);
            List<Double> yPredict = dataset.stream().map(predictor::predict).collect(Collectors.toList());
            List<Double> finalExpectedResults = expectedResults;
            List<Double> errors = IntStream.range(0, yPredict.size()).mapToDouble(v -> yPredict.get(v) - finalExpectedResults.get(v)).boxed().collect(Collectors.toList());

            Optional<Double> opt = errors.stream().mapToDouble(d -> d).boxed().max(Double::compareTo);
            if (opt.isPresent() && opt.get() != 0) {
                errors = errors.stream().map(v -> Math.pow(v / opt.get(), 2)).collect(Collectors.toList());
            }
            List<Double> finalErrors = errors;
            double trainingError = IntStream.range(0, errors.size()).mapToDouble(v -> finalErrors.get(v) * sampleWeights[v]).sum();

            // Determine amount of say for that classifier
            System.out.println("Training error: " + trainingError);
            trainingError += 0.001;
            double alpha = 0.5 * Math.log1p((1 - trainingError) / trainingError);
            trainedClassifiers.put(predictor, alpha);

            // Update data sample weights
            double beta = trainingError / (1 - trainingError);
            if (i != regressorCount - 1) {
                for (int w = 0; w < sampleWeights.length; w++) {
                    sampleWeights[w] *= Math.pow(beta, 1 - finalErrors.get(w));
                }
            }

            // Normalize data sample weights
            double sum = Arrays.stream(sampleWeights).sum();
            for (int w = 0; w < sampleWeights.length; w++) {
                sampleWeights[w] /= sum;
            }

            // Build a new dataset
            System.out.println("Building dataset:");
            System.out.println("Current weight distribution: " + Arrays.toString(sampleWeights));
            List<List<Double>> newDataset = new ArrayList<>();
            List<Double> newExpectedResults = new ArrayList<>();

            for (int j = 0; j < dataset.size(); j++) {
                double randomWeight = ThreadLocalRandom.current().nextDouble();
                int index = 0;
                while (randomWeight > sampleWeights[index]) {
                    randomWeight -= sampleWeights[index];
                    index++;
                }
                System.out.println(index);

                newDataset.add(dataset.get(index));
                newExpectedResults.add(expectedResults.get(index));
            }

            System.out.println("New expected results: " + newExpectedResults);

            dataset = newDataset;
            expectedResults = newExpectedResults;
            Arrays.fill(sampleWeights, 1.0 / ((double) dataset.size()));
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
