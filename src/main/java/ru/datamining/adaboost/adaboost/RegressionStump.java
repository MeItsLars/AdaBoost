package ru.datamining.adaboost.adaboost;
import lombok.RequiredArgsConstructor;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;


@RequiredArgsConstructor
public class RegressionStump {

    private final int attributeIndex;
    private double splitPosition, leftNode, rightNode;


    public void train(List<List<Double>> data) {
        List<Double> attributeData = data.get(attributeIndex);
        this.splitPosition = bestBinarySplit(attributeData);
    }


    public Double predict(List<Double> entry) {
        return entry.get(attributeIndex) <= splitPosition ? leftNode : rightNode;
    }


    private double bestBinarySplit(List<Double> attributeData){
        List<Double> attributeDataCopy = new LinkedList<>(attributeData).stream()
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        double minSSR = Integer.MAX_VALUE;
        double bestSplit = 0;

        for (int i = 0; i < attributeDataCopy.size() - 1; i++) {
            // TODO: add first and last split (if necessary), see slide 39 chap4a
            double splitPosition = (attributeDataCopy.get(i) + attributeDataCopy.get(i + 1)) / 2.0;

            // Compute the average of the data points smaller than or equal to the split position
            double averageLeft = attributeData.stream()
                    .filter(num -> num <= splitPosition)
                    .mapToDouble(num -> num).average().orElse(0);

            // Compute the average of the data points larger than the split position
            double averageRight = attributeData.stream()
                    .filter(num -> num > splitPosition)
                    .mapToDouble(num -> num).average().orElse(0);

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
        return bestSplit;
    }
}
