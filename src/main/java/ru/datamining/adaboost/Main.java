package ru.datamining.adaboost;

import ru.datamining.adaboost.adaboost.AdaBoost;
import ru.datamining.adaboost.adaboost.RegressionTree;
import ru.datamining.adaboost.util.CsvReader;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) {
        File file = new File("./data/student-por.csv");

        System.out.println("Preparing input data...");

        List<List<String>> classInputData = CsvReader.readCsvDataFile(file);
        classInputData = classInputData.stream().map(input -> input.stream().map(value -> value.replace("\"", "")).collect(Collectors.toList())).collect(Collectors.toList());

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
            doubleData.add(Double.parseDouble(data.get(30)));
            doubleData.add(Double.parseDouble(data.get(31)));
            doubleData.add(Double.parseDouble(data.get(32)));

            mappedInputData.add(doubleData);
        }

        List<List<Double>> dataset = new ArrayList<>();
        List<Double> expectedResults = new ArrayList<>();

        for (List<Double> data : mappedInputData) {
            dataset.add(data.subList(0, 32));
            expectedResults.add(data.get(32));
        }

        List<List<Double>> trainData = dataset.subList(0, 600);
        List<Double> trainResults = expectedResults.subList(0, 600);
        List<List<Double>> testData = dataset.subList(0, 600);
        List<Double> testResults = expectedResults.subList(0, 600);

        System.out.println("Training AdaBoost...");
        AdaBoost adaBoost = new AdaBoost(30, 3, 2);
        adaBoost.train(trainData, trainResults);

        RegressionTree regressionTree = new RegressionTree(30, 2);
        regressionTree.train(trainData, trainResults);

        System.out.println("Computing errors...");
        List<Double> errors = new ArrayList<>();
        List<Double> regressionErrors = new ArrayList<>();
        for (int i = 0; i < testData.size(); i++) {
            List<Double> testEntry = testData.get(i);
            double expectedResult = testResults.get(i);
            errors.add(Math.abs(expectedResult - adaBoost.predict(testEntry)));
            regressionErrors.add(Math.abs(expectedResult - regressionTree.predict(testEntry)));
        }

        System.out.println("Average AdaBoost error: " + errors.stream().mapToDouble(i -> i).average().getAsDouble());
        System.out.println("Average Regression Tree error: " + regressionErrors.stream().mapToDouble(i -> i).average().getAsDouble());
    }
}