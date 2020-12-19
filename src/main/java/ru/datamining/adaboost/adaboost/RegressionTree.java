package ru.datamining.adaboost.adaboost;

import java.util.*;

/**
 * Class for representing a regression tree. This is an implementation of the 'Regressor' interface.
 */
public class RegressionTree implements Regressor {

    // The minimum number of observations required for a split
    private final int minNrObservations;
    // The maximum depth of the tree before only leaves are generated
    private final int maxDept;
    // The root binary node
    private BinaryNode rootNode;

    public RegressionTree(int maxDept, int minNrObservations) {
        this.maxDept = maxDept;
        this.minNrObservations = minNrObservations;
    }

    /**
     * Trains the regression tree on the given input data
     *
     * @param data            The input data, formatted into a list of data entries
     * @param expectedResults The expected results, formatted into a list of values
     */
    public void train(List<List<Double>> data, List<Double> expectedResults) {
        this.rootNode = new BinaryNode();
        train(data, expectedResults, rootNode);
    }

    /**
     * Recursively trains the regression tree on the given input data, where the given binary node is the current node
     *
     * @param data            The input data, formatted into a list of data entries
     * @param expectedResults The expected results, formatted into a list of values
     * @param node            The current binary node
     */
    public void train(List<List<Double>> data, List<Double> expectedResults, BinaryNode node) {
        // Computes the average of the values in the current node, used for when the node becomes a leaf
        node.setAverage(expectedResults.stream().mapToDouble(v -> v).average().orElse(0));

        // If the node depth is bigger than or equal to the max depth, turn this node into a leaf
        if (node.getLevel() >= maxDept) {
            return;
        }

        // Compute the best possible split on the current node
        node.computeBestSplit(data, expectedResults);

        // Create lists of data that have to be considered in the left and right child nodes
        List<List<Double>> leftNodeData = new ArrayList<>();
        List<Double> leftNodeResults = new ArrayList<>();
        List<List<Double>> rightNodeData = new ArrayList<>();
        List<Double> rightNodeResults = new ArrayList<>();

        // Fill the lists of data according to the previously made split on the node
        for (int i = 0; i < data.size(); i++) {
            List<Double> entry = data.get(i);

            if (entry.get(node.getAttributeIndex()) <= node.getSplitPosition()) {
                leftNodeData.add(entry);
                leftNodeResults.add(expectedResults.get(i));
            } else {
                rightNodeData.add(entry);
                rightNodeResults.add(expectedResults.get(i));
            }
        }

        // If either of the two sets is empty, this current node becomes a leaf
        if (leftNodeData.isEmpty() || rightNodeData.isEmpty()) {
            return;
        }

        // Create the new nodes
        BinaryNode leftNode = new BinaryNode(node);
        BinaryNode rightNode = new BinaryNode(node);

        node.setLeftChild(leftNode);
        node.setRightChild(rightNode);

        // If there are still enough elements in the left node, recursively train it.
        // Otherwise, make the left node a leaf.
        if (leftNodeData.size() >= minNrObservations) {
            train(leftNodeData, leftNodeResults, leftNode);
        } else {
            leftNode.setAverage(leftNodeResults.stream().mapToDouble(d -> d).average().orElse(Integer.MIN_VALUE));
        }

        // If there are still enough elements in the right node, recursively train it.
        // Otherwise, make the right node a leaf.
        if (rightNodeData.size() >= minNrObservations) {
            train(rightNodeData, rightNodeResults, rightNode);
        } else {
            rightNode.setAverage(rightNodeResults.stream().mapToDouble(d -> d).average().orElse(Integer.MIN_VALUE));
        }
    }

    /**
     * Predicts the output value of the given input entry via the regression tree
     *
     * @param entry The input entry
     * @return The predicted output
     */
    public double predict(List<Double> entry) {
        BinaryNode currentNode = rootNode;
        // Recursively follow the binary nodes until a leaf is found
        while (true) {
            if (entry.get(currentNode.getAttributeIndex()) <= currentNode.getSplitPosition() && currentNode.getLeftChild() != null) {
                currentNode = currentNode.getLeftChild();
            } else if (currentNode.getRightChild() != null) {
                currentNode = currentNode.getRightChild();
            } else return currentNode.getAverage();
        }
    }
}