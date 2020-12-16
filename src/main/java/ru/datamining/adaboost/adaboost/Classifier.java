package ru.datamining.adaboost.adaboost;

import java.util.List;

public interface Classifier {

    enum ClassificationType { NUMERICAL, NOMINAL }

    void train(List<List<String>> data, List<String> expectedResult);

    String predict(List<String> entry);

}
