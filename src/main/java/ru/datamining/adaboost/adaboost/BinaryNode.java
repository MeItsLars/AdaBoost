package ru.datamining.adaboost.adaboost;

import lombok.Getter;
import lombok.Setter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Class for representing a binary node, used in regression trees.
 */
@Getter
@Setter
public class BinaryNode {

    // The average of the values in this node
    private double average;
    // After computing the best split: the current split position
    private double splitPosition;
    // After computing the best split: the current sum of squared errors
    private double sumSquaredError = Integer.MAX_VALUE;
    // The depth of this binary node in the regression tree
    private final int level;
    // After computing the best split: the attribute index of the attribute that was split on
    private int attributeIndex;
    // The left child binary node
    private BinaryNode leftChild;
    // The right child binary node
    private BinaryNode rightChild;
    // The parent binary node
    private BinaryNode parent;

    public BinaryNode(BinaryNode parent) {
        this.parent = parent;
        this.level = parent.getLevel() + 1;
    }

    public BinaryNode() {
        this.parent = null;
        this.level = 0;
    }

    /**
     * Computes the best possible split by checking all possible attribute splits, and choosing the one with
     * the lowest sum of squared error value.
     *
     * @param data    The input data
     * @param results The expected results
     */
    public void computeBestSplit(List<List<Double>> data, List<Double> results) {
        // Loop through all attributes
        for (int i = 0; i < data.get(0).size(); i++) {
            int finalI = i;
            // Create a column value of the current attribute index
            List<Double> attributeData = data.stream().map(list -> list.get(finalI)).collect(Collectors.toList());
            // Compute the map containing all split values. The map maps SSR's to split indices
            Map<Double, Double> splitMap = computeSplits(attributeData, results);
            // Choose the lowest SSR
            double ssr = splitMap.keySet().stream().mapToDouble(v -> v).min().orElse(Integer.MAX_VALUE);
            // If it was lower than the current SSR, set the current SSR, split, and attribute index
            if (ssr < sumSquaredError) {
                this.sumSquaredError = ssr;
                this.splitPosition = splitMap.get(ssr);
                this.attributeIndex = i;
            }
        }
    }

    /**
     * Computes a map that maps SSR values to split indices for all possible splits on the given attribute value column.
     *
     * @param attributeData The attribute value column
     * @param results       The set of expected results
     * @return The resulting map
     */
    private Map<Double, Double> computeSplits(List<Double> attributeData, List<Double> results) {
        // Computes all different values, distinct and sorted
        List<Double> attributeDataCopy = attributeData.stream().distinct().sorted().collect(Collectors.toList());
        attributeDataCopy.add(0, attributeDataCopy.get(0) - 1);
        attributeDataCopy.add(attributeDataCopy.get(attributeDataCopy.size() - 1) + 1);

        Map<Double, Double> splitMap = new HashMap<>();

        // Loop through all values
        for (int i = 0; i < attributeDataCopy.size() - 1; i++) {
            // Create a split index by taking the average of the current value and the next value
            double currentSplitValue = (attributeDataCopy.get(i) + attributeDataCopy.get(i + 1)) / 2.0;

            // Compute the average of the data points smaller than or equal to the split position
            // and the average of the data point larger than the split position
            double leftCount = 0;
            double rightCount = 0;
            double totalLeft = 0;
            double totalRight = 0;

            for (int j = 0; j < results.size(); j++) {
                if (attributeData.get(j) <= currentSplitValue) {
                    totalLeft += results.get(j);
                    leftCount++;
                } else {
                    totalRight += results.get(j);
                    rightCount++;
                }
            }

            double averageLeft = leftCount == 0 ? 0 : totalLeft / leftCount;
            double averageRight = rightCount == 0 ? 0 : totalRight / rightCount;

            // Compute sum of squared residuals for current split
            double ssr = 0;
            for (int j = 0; j < attributeData.size(); j++) {
                if (attributeData.get(j) <= currentSplitValue) {
                    ssr += Math.pow(results.get(j) - averageLeft, 2);
                } else {
                    ssr += Math.pow(results.get(j) - averageRight, 2);
                }
            }
            // Add the result to the map
            splitMap.put(ssr, currentSplitValue);
        }
        return splitMap;
    }
}
