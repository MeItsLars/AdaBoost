package ru.datamining.adaboost.adaboost;

import org.jdmp.core.util.DefaultObservableMap;

import java.lang.reflect.Type;
import java.util.*;
import java.util.stream.Collectors;

public class BinaryNode{
    private double average, splitPosition;
    private double sumSquaredError;
    private final int level;
    private int attributeIndex;
    BinaryNode leftChild, rightChild, parent;


    public BinaryNode(BinaryNode parent){
        this.parent = parent;
        this.level = parent.getLevel() + 1;
    }


    public BinaryNode(){
        this.parent = null;
        this.level = 0;
    }


    public void computeBestSplit(List<List<Double>> data, List<Double> results) {
        // Compute best attribute to split on and its best split position,
        // based on the minimum sum of squared residual of each of the attributes
        double bestSsr = Integer.MAX_VALUE;
        for (int i = 0; i < data.get(0).size(); i++) {
            int finalI = i;
            List<Double> attributeData = data.stream().map(list -> list.get(finalI)).collect(Collectors.toList());
            Map<Double, Double> splitMap = computeSplits(attributeData, results);
            double ssr = splitMap.keySet().stream().mapToDouble(v -> v).min().orElse(Integer.MAX_VALUE);
            if (ssr < bestSsr) {
                bestSsr = ssr;
                this.sumSquaredError = ssr;
                this.splitPosition = splitMap.get(ssr);
                this.attributeIndex = i;
            }
        }
    }


    private Map<Double, Double> computeSplits(List<Double> attributeData, List<Double> results){
        // Computes all possible splits with their sum  squared residuals
        List<Double> attributeDataCopy = attributeData.stream().distinct().sorted().collect(Collectors.toList());
        attributeDataCopy.add(0, attributeDataCopy.get(0) - 1);
        attributeDataCopy.add(attributeDataCopy.get(attributeDataCopy.size() -1) +1);

        Map<Double, Double> splitMap = new HashMap<>();

        for (int i = 0; i < attributeDataCopy.size() - 1; i++) {
            double currentSplitPosition = (attributeDataCopy.get(i) + attributeDataCopy.get(i + 1)) / 2.0;

            // Compute the average of the data points smaller than or equal to the split position
            // and the average of the data point larger than the split position
            double leftCount = 0;
            double rightCount = 0;
            double totalLeft = 0;
            double totalRight = 0;

            for (int j = 0; j < results.size(); j++) {
                if (attributeData.get(j) <= currentSplitPosition) {
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
                if (attributeData.get(j) <= currentSplitPosition) {
                    ssr += Math.pow(results.get(j) - averageLeft, 2);
                } else {
                    ssr += Math.pow(results.get(j) - averageRight, 2);
                }
            }
            splitMap.put(ssr, currentSplitPosition);
        }
        System.out.println(splitMap);
        return splitMap;
    }


    public void setLeftChild(BinaryNode leftChild) {
        this.leftChild = leftChild;
    }


    public void setRightChild(BinaryNode rightChild) {
        this.rightChild = rightChild;
    }

    public BinaryNode getLeftChild() {
        return leftChild;
    }

    public BinaryNode getRightChild() {
        return rightChild;
    }

    public double getAverage() {
        return average;
    }

    public void setAverage(double average) {
        this.average = average;
    }

    public int getLevel(){
        return level;
    }


    public int getAttributeIndex() {
        return attributeIndex;
    }


    public double getSplitPosition() {
        return splitPosition;
    }


    @Override
    public String toString() {
        return "BinaryNode{" +
                "average=" + average +
                ", splitPosition=" + splitPosition +
                ", sumSquaredError=" + sumSquaredError +
                ", level=" + level +
                ", attributeIndex=" + attributeIndex +
                '}';
    }

}
