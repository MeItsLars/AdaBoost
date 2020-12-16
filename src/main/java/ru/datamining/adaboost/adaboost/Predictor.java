package ru.datamining.adaboost.adaboost;

import java.util.List;

public interface Predictor {

    void train(List<List<String>> data, List<String> expectedResult);

    String predict(List<String> entry);

}
