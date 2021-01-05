package ru.datamining.adaboost;

import lombok.SneakyThrows;
import ru.datamining.adaboost.adaboost.AdaBoostRegressor;
import ru.datamining.adaboost.adaboost.RegressionTree;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Main {

    // The output logger
    private static final Logger LOGGER = Logger.getLogger("AdaBoost");

    public static void main(String[] args) {
        LOGGER.info("Preparing input data...");
        File file = new File("./data/forestfires.csv");

        // Read the data value through a CSV file
        List<List<String>> classInputData = readCsvDataFile(file);

        // Create the dataset associated nominal values
        List<String> months = Arrays.asList("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec");
        List<String> days = Arrays.asList("mon", "tue", "wed", "thu", "fri", "sat", "sun");

        // Map the input string values which may be both nominal and numerical, to purely numerical values
        List<List<Double>> mappedInputData = new ArrayList<>();
        for (int i = 1; i < classInputData.size(); i++) {
            List<String> data = classInputData.get(i);

            List<Double> doubleData = new ArrayList<>();
            for (String value : data) {
                try {
                    // Check if the input value can be parsed as a double
                    double d = Double.parseDouble(value);
                    doubleData.add(d);
                } catch (NumberFormatException e) {
                    // If not, check if it is a month value
                    double d = months.indexOf(value);

                    if (d >= 0) {
                        doubleData.add(d);
                    } else {
                        // If not, check if it is a day value
                        d = days.indexOf(value);

                        if (d >= 0) {
                            doubleData.add(d);
                        } else {
                            // If not, there is an error in the input data
                            throw new IllegalArgumentException("Wrong format: " + value);
                        }
                    }
                }
            }

            mappedInputData.add(doubleData);
        }

        List<List<Double>> dataset = new ArrayList<>();
        List<Double> results = new ArrayList<>();

        // We split the input data into a dataset and expected results
        for (List<Double> data : mappedInputData) {
            dataset.add(data.subList(0, 12));
            results.add(data.get(12));
        }

        // We use K-Fold Cross-Validation:
        int splits = 10;
        int splitSize = dataset.size() / splits;
        LOGGER.log(Level.INFO, "Applying K-Fold Cross-Validation with {0} folds and a split size of {1}...", new Object[]{splits, splitSize});
        for (int k = 0; k < splits; k++) {
            // Create the train and test datasets:
            List<List<Double>> trainX = Stream.concat(dataset.subList(0, k * splitSize).stream(), dataset.subList((k + 1) * splitSize, splits * splitSize).stream()).collect(Collectors.toList());
            List<Double> trainY = Stream.concat(results.subList(0, k * splitSize).stream(), results.subList((k + 1) * splitSize, splits * splitSize).stream()).collect(Collectors.toList());
            List<List<Double>> testX = dataset.subList(k * splitSize, (k + 1) * splitSize);
            List<Double> testY = results.subList(k * splitSize, (k + 1) * splitSize);

            LOGGER.log(Level.INFO, "Iteration {0}", k);
            LOGGER.info("Training AdaBoost and a regression tree...");
            // We create an AdaBoost regressor and train it on the data
            AdaBoostRegressor adaBoostRegressor = new AdaBoostRegressor(30, 3, 2);
            adaBoostRegressor.train(trainX, trainY);

            // We create a regression tree and train it on the data
            RegressionTree regressionTree = new RegressionTree(30, 2);
            regressionTree.train(trainX, trainY);

            LOGGER.info("Computing mean-squared-error and standard deviation for the test data...");
            // Predict all test data on AdaBoost and a regression tree:
            List<Double> adaBoostResults = new ArrayList<>();
            List<Double> regressionResults = new ArrayList<>();
            for (List<Double> testEntry : testX) {
                adaBoostResults.add(adaBoostRegressor.predict(testEntry));
                regressionResults.add(regressionTree.predict(testEntry));
            }

            // Compute standard deviation (STD):
            double adaBoostMean = adaBoostResults.stream().mapToDouble(d -> d).average().orElseThrow();
            double regressionMean = regressionResults.stream().mapToDouble(d -> d).average().orElseThrow();
            double adaBoostSTD = Math.sqrt((1.0 / (double) trainX.size()) * adaBoostResults.stream().mapToDouble(d -> Math.pow(d - adaBoostMean, 2)).sum());
            double regressionSTD = Math.sqrt((1.0 / (double) trainX.size()) * regressionResults.stream().mapToDouble(d -> Math.pow(d - regressionMean, 2)).sum());
            LOGGER.log(Level.INFO, "AdaBoost STD: {0}, Regression Tree STD: {1}", new Object[]{adaBoostSTD, regressionSTD});

            // Compute mean squared error (MSE)
            double adaBoostMSE = IntStream.range(0, testX.size()).mapToDouble(i -> Math.pow(adaBoostResults.get(i) - testY.get(i), 2)).average().orElseThrow();
            double regressionMSE = IntStream.range(0, testX.size()).mapToDouble(i -> Math.pow(regressionResults.get(i) - testY.get(i), 2)).average().orElseThrow();
            LOGGER.log(Level.INFO, "AdaBoost MSE: {0}, Regression Tree MSE: {1}", new Object[]{adaBoostMSE, regressionMSE});
        }
    }

    /**
     * Reads a .csv dataset into a list of entries
     *
     * @param file The input file
     * @return The list of entries
     */
    @SneakyThrows
    private static List<List<String>> readCsvDataFile(File file) {
        return Files.readAllLines(file.toPath()).stream()
                .map(input -> Arrays.asList(input.split(",")))
                .collect(Collectors.toList());
    }
}