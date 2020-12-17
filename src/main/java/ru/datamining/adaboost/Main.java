package ru.datamining.adaboost;

import ru.datamining.adaboost.adaboost.AdaBoost;
import ru.datamining.adaboost.adaboost.RegressionStump;
import ru.datamining.adaboost.util.CsvReader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        File file = new File("./data/student-por.csv");

        System.out.println("Preparing input data...");

        List<List<String>> classInputData = CsvReader.readCsvDataFile(file);

        List<String> jobs = Arrays.asList("teacher", "health", "services", "at_home", "other");
        List<String> reason = Arrays.asList("home", "reputation", "course", "other");
        List<String> guardian = Arrays.asList("mother", "father", "other");

        List<List<Double>> mappedInputData = new ArrayList<>();
        for (int i = 1; i < classInputData.size(); i++) {
            List<String> data = classInputData.get(i);

            List<Double> doubleData = new ArrayList<>();
            doubleData.add(data.get(0).equals("GP") ? 0.0 : 1.0);
            doubleData.add(data.get(1).equals("M") ? 0.0 : 1.0);
            doubleData.add(Double.parseDouble(data.get(2)));
            doubleData.add(data.get(3).equals("U") ? 0.0 : 1.0);
            doubleData.add(data.get(4).equals("LE3") ? 0.0 : 1.0);
            doubleData.add(data.get(5).equals("T") ? 0.0 : 1.0);
            doubleData.add(Double.parseDouble(data.get(6)));
            doubleData.add(Double.parseDouble(data.get(7)));
            doubleData.add((double) jobs.indexOf(data.get(8)));
            doubleData.add((double) jobs.indexOf(data.get(9)));
            doubleData.add((double) reason.indexOf(data.get(10)));
            doubleData.add((double) guardian.indexOf(data.get(11)));
            doubleData.add(Double.parseDouble(data.get(12)));
            doubleData.add(Double.parseDouble(data.get(13)));
            doubleData.add(Double.parseDouble(data.get(14)));
            doubleData.add(data.get(15).equals("no") ? 0.0 : 1.0);
            doubleData.add(data.get(16).equals("no") ? 0.0 : 1.0);
            doubleData.add(data.get(17).equals("no") ? 0.0 : 1.0);
            doubleData.add(data.get(18).equals("no") ? 0.0 : 1.0);
            doubleData.add(data.get(19).equals("no") ? 0.0 : 1.0);
            doubleData.add(data.get(20).equals("no") ? 0.0 : 1.0);
            doubleData.add(data.get(21).equals("no") ? 0.0 : 1.0);
            doubleData.add(data.get(22).equals("no") ? 0.0 : 1.0);
            doubleData.add(Double.parseDouble(data.get(23)));
            doubleData.add(Double.parseDouble(data.get(24)));
            doubleData.add(Double.parseDouble(data.get(25)));
            doubleData.add(Double.parseDouble(data.get(26)));
            doubleData.add(Double.parseDouble(data.get(27)));
            doubleData.add(Double.parseDouble(data.get(28)));
            doubleData.add(Double.parseDouble(data.get(29)));
            doubleData.add(Double.parseDouble(data.get(30).substring(1, data.get(30).length() - 1)));
            doubleData.add(Double.parseDouble(data.get(31).substring(1, data.get(31).length() - 1)));
            doubleData.add(Double.parseDouble(data.get(32)));

            mappedInputData.add(doubleData);
        }

        List<List<Double>> dataset = new ArrayList<>();
        List<Double> expectedResults = new ArrayList<>();

        for (List<Double> data : mappedInputData) {
            dataset.add(data.subList(0, 32));
            expectedResults.add(data.get(32));
        }

        List<List<Double>> trainData = dataset.subList(0, dataset.size());
        List<Double> trainResults = expectedResults.subList(0, expectedResults.size());
        List<List<Double>> testData = dataset.subList(0, dataset.size());
        List<Double> testResults = expectedResults.subList(0, expectedResults.size());

        System.out.println("Training AdaBoost...");
        AdaBoost adaBoost = new AdaBoost();
        adaBoost.train(trainData, trainResults);

        System.out.println("Computing errors...");
        List<Double> errors = new ArrayList<>();
        for (int i = 0; i < testData.size(); i++) {
            List<Double> testEntry = testData.get(i);
            double expectedResult = testResults.get(i);
            errors.add(Math.abs(expectedResult - adaBoost.predict(testEntry)));
        }

        System.out.println("Average error: " + errors.stream().mapToDouble(i -> i).average().getAsDouble());

        /*List<List<Double>> dataset = new ArrayList<>();

        dataset.add(Arrays.asList(19.0, 59.0, 185.0));
        dataset.add(Arrays.asList(22.0, 78.0, 188.0));
        dataset.add(Arrays.asList(30.0, 70.0, 180.0));
        dataset.add(Arrays.asList(40.0, 62.0, 168.0));
        dataset.add(Arrays.asList(50.0, 73.0, 177.0));
        dataset.add(Arrays.asList(60.0, 100.0, 210.0));
        dataset.add(Arrays.asList(70.0, 70.0, 183.0));
        dataset.add(Arrays.asList(80.0, 65.0, 173.0));
        dataset.add(Arrays.asList(90.0, 55.0, 150.0));
        dataset.add(Arrays.asList(100.0, 60.0, 167.0));

        List<List<Double>> data = new ArrayList<>();
        List<Double> expectedResult = new ArrayList<>();

        for (List<Double> entry : dataset) {
            data.add(Arrays.asList(entry.get(0), entry.get(1)));
            expectedResult.add(entry.get(2));
        }

        AdaBoost adaBoost = new AdaBoost();
        adaBoost.train(data, expectedResult);

        double result = adaBoost.predict(Arrays.asList(60.0, 100.0));

        System.out.println("Deze tantoe oude man is: " + result + "cm");*/
    }

}