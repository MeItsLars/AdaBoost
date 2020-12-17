package ru.datamining.adaboost.adaboost;

import java.util.List;

public interface Predictor {

    void train(List<List<Double>> data, List<Double> expectedResult);

    double predict(List<Double> entry);

}