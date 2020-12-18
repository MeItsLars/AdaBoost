package ru.datamining.adaboost.adaboost;

import java.util.*;

public class RegressionTree implements Predictor {

    private final int minNrObservations;
    private final int maxDept;
    private BinaryNode rootNode;

    public RegressionTree(int maxDept, int minNrObservations) {
        this.maxDept = maxDept;
        this.minNrObservations = minNrObservations;
    }

    public void train(List<List<Double>> data, List<Double> results) {
        this.rootNode = new BinaryNode();
        train(data, results, rootNode);
    }


    public void train(List<List<Double>> data, List<Double> results, BinaryNode node) {
        node.setAverage(results.stream().mapToDouble(v -> v).average().orElse(0));

        if (node.getLevel() >= maxDept) {
            return;
        }

        node.computeBestSplit(data, results);

        // Create lists of data that has to be considered in the left and right child nodes
        List<List<Double>> leftNodeData = new ArrayList<>();
        List<Double> leftNodeResults = new ArrayList<>();
        List<List<Double>> rightNodeData = new ArrayList<>();
        List<Double> rightNodeResults = new ArrayList<>();

        for (int i = 0; i < data.size(); i++) {
            List<Double> entry = data.get(i);

            if (entry.get(node.getAttributeIndex()) <= node.getSplitPosition()) {
                leftNodeData.add(entry);
                leftNodeResults.add(results.get(i));
            } else {
                rightNodeData.add(entry);
                rightNodeResults.add(results.get(i));
            }
        }

        if (leftNodeData.isEmpty() || rightNodeData.isEmpty()) {
            return;
        }

        BinaryNode leftNode = new BinaryNode(node);
        BinaryNode rightNode = new BinaryNode(node);

        node.setLeftChild(leftNode);
        node.setRightChild(rightNode);

        if (leftNodeData.size() >= minNrObservations) {
            train(leftNodeData, leftNodeResults, leftNode);
        } else {
            leftNode.setAverage(leftNodeResults.stream().mapToDouble(d -> d).average().orElse(Integer.MIN_VALUE));
        }

        if (rightNodeData.size() >= minNrObservations) {
            train(rightNodeData, rightNodeResults, rightNode);
        } else {
            rightNode.setAverage(rightNodeResults.stream().mapToDouble(d -> d).average().orElse(Integer.MIN_VALUE));
        }
    }


    public double predict(List<Double> entry) {
        BinaryNode currentNode = rootNode;
        while (true) {
            if (entry.get(currentNode.getAttributeIndex()) <= currentNode.getSplitPosition() && currentNode.getLeftChild() != null) {
                currentNode = currentNode.getLeftChild();
            } else if (currentNode.getRightChild() != null) {
                currentNode = currentNode.getRightChild();
            } else return currentNode.getAverage();
        }
    }
}