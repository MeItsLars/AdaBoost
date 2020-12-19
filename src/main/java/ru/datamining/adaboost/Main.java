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
        List<Double> expectedResults = new ArrayList<>();

        // We split the input data into a dataset and expected results
        for (List<Double> data : mappedInputData) {
            dataset.add(data.subList(0, 12));
            expectedResults.add(data.get(12));
        }

        // We split the data and results into train and test data
        List<List<Double>> trainData = dataset.subList(0, 300);
        List<Double> trainResults = expectedResults.subList(0, 300);
        List<List<Double>> testData = dataset.subList(300, 500);
        List<Double> testResults = expectedResults.subList(300, 500);

        LOGGER.info("Training AdaBoost...");
        // We create an AdaBoost regressor and train it on the data
        AdaBoostRegressor adaBoostRegressor = new AdaBoostRegressor(30, 3, 2);
        adaBoostRegressor.train(trainData, trainResults);

        // We create a regression tree and train it on the data
        RegressionTree regressionTree = new RegressionTree(30, 2);
        regressionTree.train(trainData, trainResults);

        LOGGER.info("Computing error values...");
        // We compute the error values for all test data for both AdaBoost and regression tree
        List<Double> adaBoostErrors = new ArrayList<>();
        List<Double> regressionErrors = new ArrayList<>();
        for (int i = 0; i < testData.size(); i++) {
            List<Double> testEntry = testData.get(i);
            double expectedResult = testResults.get(i);
            adaBoostErrors.add(Math.abs(expectedResult - adaBoostRegressor.predict(testEntry)));
            regressionErrors.add(Math.abs(expectedResult - regressionTree.predict(testEntry)));
        }

        // Print the resulting errors
        LOGGER.log(Level.INFO, "Average AdaBoost error: {0}", adaBoostErrors.stream().mapToDouble(i -> i).average().orElseThrow());
        LOGGER.log(Level.INFO, "Average Regression Tree error: {0}", regressionErrors.stream().mapToDouble(i -> i).average().orElseThrow());
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