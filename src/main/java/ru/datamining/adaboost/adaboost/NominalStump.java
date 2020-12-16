package ru.datamining.adaboost.adaboost;

import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RequiredArgsConstructor
public class NominalStump implements Classifier {

    private final int attributeIndex;
    private final ClassificationType classificationType;

    private final Map<String, String> trainResults = new HashMap<>();

    @Override
    public void train(List<List<String>> data, List<String> expectedResult) {
        List<String> attribute = data.get(attributeIndex);

        switch (classificationType) {
            case NOMINAL: {
                Map<String, List<String>> countedValues = new HashMap<>();
                int index = 0;
                for (String value : attribute) {
                    countedValues.computeIfAbsent(value, i -> new ArrayList<>()).add(expectedResult.get(index));
                    index++;
                }
                countedValues.forEach((key, list) -> trainResults.put(key, mostCommonValue(list)));
                break;
            }
            case NUMERICAL:
                // Write in report: Decision to use average
                Map<String, List<Double>> countedValues = new HashMap<>();
                int index = 0;
                for (String value : attribute) {
                    countedValues.computeIfAbsent(value, i -> new ArrayList<>()).add(Double.valueOf(expectedResult.get(index)));
                    index++;
                }
                countedValues.forEach((value, numbers) -> {
                    trainResults.put(value, String.valueOf(numbers.stream().mapToDouble(Double::valueOf).average().orElse(0)));
                });
                break;
            default:
                break;
        }
    }

    private <T> T mostCommonValue(List<T> values) {
        Map<T, Integer> occurrences = new HashMap<>();
        values.forEach(value -> occurrences.put(value, occurrences.getOrDefault(value, 0)));

        T result = null;
        int highestCount = 0;

        for (Map.Entry<T, Integer> entry : occurrences.entrySet()) {
            if (entry.getValue() > highestCount) {
                highestCount = entry.getValue();
                result = entry.getKey();
            }
        }

        return result;
    }

    @Override
    public String predict(List<String> entry) {
        return trainResults.get(entry.get(attributeIndex));
    }
}
