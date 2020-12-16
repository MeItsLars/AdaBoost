package ru.datamining.adaboost.adaboost;

import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RequiredArgsConstructor
public class NumericalStump implements Classifier {

    private final int attributeIndex;
    private final ClassificationType classificationType;

    @Override
    public void train(List<List<String>> data, List<String> expectedResult) {
        List<Double> attribute = data.get(attributeIndex).stream().mapToDouble(Double::valueOf).boxed().collect(Collectors.toList());

        switch (classificationType) {
            case NOMINAL:
                break;
            case NUMERICAL:
                break;
            default:
                break;
        }
    }

    @Override
    public String predict(List<String> entry) {
        return null;
    }
}
