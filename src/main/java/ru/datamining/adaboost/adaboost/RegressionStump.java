package ru.datamining.adaboost.adaboost;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

@RequiredArgsConstructor
@Getter
public class RegressionStump implements Predictor {

    private final int attributeIndex;
    private double splitPosition;
    private double leftNode;
    private double rightNode;

    @Override
    public void train(List<List<Double>> data, List<Double> expectedResults) {
        List<Double> attributeData = data.stream().map(list -> list.get(attributeIndex)).collect(Collectors.toList());

        List<Double> attributeDataCopy = new LinkedList<>(attributeData).stream()
                .distinct()
                .sorted()
                .collect(Collectors.toList());
        attributeDataCopy.set(0, attributeDataCopy.get(0) - 1);
        attributeDataCopy.add(attributeDataCopy.get(attributeDataCopy.size() - 1) + 1);

        double minSSR = Integer.MAX_VALUE;
        double bestSplit = 0;

        for (int i = 0; i < attributeDataCopy.size() - 1; i++) {
            double currentSplitPosition = (attributeDataCopy.get(i) + attributeDataCopy.get(i + 1)) / 2.0;

            // Compute the average of the data points smaller than or equal to the split position
            // and the average of the data point larger than the split position
            double leftCount = 0;
            double rightCount = 0;
            double totalLeft = 0;
            double totalRight = 0;

            for (int j = 0; j < expectedResults.size(); j++) {
                if (attributeData.get(j) <= currentSplitPosition) {
                    totalLeft += expectedResults.get(j);
                    leftCount++;
                } else {
                    totalRight += expectedResults.get(j);
                    rightCount++;
                }
            }

            double averageLeft = leftCount == 0 ? 0 : totalLeft / leftCount;
            double averageRight = rightCount == 0 ? 0 : totalRight / rightCount;

            // Compute sum of squared residuals for current split
            double ssr = 0;
            for (Double attribute : attributeData) {
                if (attribute <= currentSplitPosition) {
                    ssr += Math.pow(attribute - averageLeft, 2);
                } else {
                    ssr += Math.pow(attribute - averageRight, 2);
                }
            }

            // Check if current split is better than previous best split
            // update minSSR, bestSplit leftNode and rightNode
            if (ssr < minSSR) {
                minSSR = ssr;
                bestSplit = currentSplitPosition;
                leftNode = averageLeft;
                rightNode = averageRight;
            }
        }
        splitPosition = bestSplit;
        System.out.println("Index: " + attributeIndex + ", left: " + leftNode + ", right: " + rightNode + ", split: " + splitPosition);
    }

    @Override
    public double predict(List<Double> entry) {
        return entry.get(attributeIndex) <= splitPosition ? leftNode : rightNode;
    }
}
