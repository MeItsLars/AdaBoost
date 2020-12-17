package ru.datamining.adaboost.adaboost;
import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;


@RequiredArgsConstructor
public class RegressionStump {

    private final int attributeIndex;
    private double splitPosition, leftNode, rightNode;


    public void train(List<List<Double>> data, List<Double> expectedResults) {
        List<Double> attributeData = data.get(attributeIndex);
        this.splitPosition = bestBinarySplit(attributeData, expectedResults);
    }


    public Double predict(List<Double> entry) {
        return entry.get(attributeIndex) <= splitPosition ? leftNode : rightNode;
    }


    private double bestBinarySplit(List<Double> attributeData, List<Double> expectedResults){
        List<Double> attributeDataCopy = new LinkedList<>(attributeData).stream()
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        double minSSR = Integer.MAX_VALUE;
        double bestSplit = 0;

        for (int i = 0; i < attributeDataCopy.size() - 1; i++) {
            // TODO: add first and last split (if necessary), see slide 39 chap4a
            double splitPosition = (attributeDataCopy.get(i) + attributeDataCopy.get(i + 1)) / 2.0;

            // Compute the average of the data points smaller than or equal to the split position;
            // and the average of the data point larger than the split position
            double leftCount = 0;
            double rightCount = 0;
            double totalLeft = 0 ;
            double totalRight = 0;

            for (int j = 0; j < expectedResults.size(); j++) {
                if (attributeData.get(j) <= splitPosition) {
                    totalLeft += expectedResults.get(j);
                    leftCount++;
                }
                else {
                    totalRight += expectedResults.get(j);
                    rightCount++;
                }
            }

            double averageLeft = totalLeft / leftCount;
            double averageRight = totalRight / rightCount;

            // Compute sum of squared residuals for current split
            double ssr = 0;
            for (Double attribute : attributeData){
                if (attribute <= splitPosition) {
                    ssr += Math.pow(attribute - averageLeft, 2);
                }
                else {
                    ssr += Math.pow(attribute - averageRight, 2);
                }
            }

            // Check if current split is better than previous best split;
            // update minSSR, bestSplit leftNode and rightNode
            if (ssr < minSSR) {
                minSSR = ssr;
                bestSplit = splitPosition;
                leftNode = averageLeft;
                rightNode = averageRight;
            }
        }

        System.out.println(bestSplit + "\t" + leftNode + "\t" + rightNode);

        return bestSplit;
    }

    public static void main(String[] args){
        List<Double> X = Arrays.asList(1.0,2.0,3.0,4.0);
        List<Double> y = Arrays.asList(1.0,0.0,1.0,1.0);
        RegressionStump rs = new RegressionStump(0);
        rs.bestBinarySplit(X,y);
    }
}
