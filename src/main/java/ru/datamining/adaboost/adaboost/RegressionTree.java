package ru.datamining.adaboost.adaboost;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

public class RegressionTree implements Predictor {

    private final int minNrObservations;
    private final int maxDept;
    BinaryNode rootNode = new BinaryNode();

    List<BinaryNode> nodes = new LinkedList<>();

    public RegressionTree(int maxDept, int minNrObservations) {
        this.minNrObservations = maxDept;
        this.maxDept = minNrObservations;
    }

    public void train(List<List<Double>> data, List<Double> results) {
        BinaryNode rootNode = new BinaryNode();
        train(data, results, rootNode);
    }


    public void train(List<List<Double>> data, List<Double> results, BinaryNode node) {
        node.setAverage(results.stream().mapToDouble(v -> v).average().orElse(0));

        if (node.getLevel() > maxDept) {
            return;
        }

        node.computeBestSplit(data, results);

        // Create lists of data that has to be considered in the left and right child nodes
        List<List<Double>> leftNodeData = data.stream()
                .filter(lst -> lst.get(node.getAttributeIndex()) <= node.getSplitPosition())
                .collect(Collectors.toList());
        List<List<Double>> rightNodeData = data.stream()
                .filter(lst -> lst.get(node.getAttributeIndex()) > node.getSplitPosition())
                .collect(Collectors.toList());

        // TODO: Create sublist of result data that has to be considered in the left and right child nodes
        List<Double> leftNodeResults = new ArrayList<>();
        List<Double> rightNodeResults = new ArrayList<>();

        BinaryNode leftNode = new BinaryNode(node);
        BinaryNode rightNode = new BinaryNode(node);

        node.setLeftChild(leftNode);
        node.setRightChild(rightNode);

        nodes.add(node);
        nodes.add(leftNode);
        nodes.add(rightNode);

        if (leftNodeData.size() > minNrObservations) {
            train(leftNodeData, leftNodeResults, leftNode);
        }
        else if (rightNodeData.size() > minNrObservations) {
            train(rightNodeData, rightNodeResults, rightNode);
        }
    }


    public double predict(List<Double> entry) {
        BinaryNode currentNode = rootNode;
        while (true) {
            if (entry.get(currentNode.getAttributeIndex()) <= rootNode.getSplitPosition() && currentNode.getLeftChild() != null) {
                currentNode = currentNode.getLeftChild();
            } else if (currentNode.getRightChild() != null) {
                currentNode = currentNode.getRightChild();
            } else return currentNode.getAverage();
        }
    }


    public static void main(String[] args) {
        RegressionTree rt = new RegressionTree(2,20);
        List<List<Double>> lst = new ArrayList<>();
        lst.add(Arrays.asList(1.0, 3.0, 4.0));
        lst.add(Arrays.asList(5.0, 6.0, 7.0));
        lst.add(Arrays.asList(8.0, 9.0, 10.0));
        lst.add(Arrays.asList(10.0, 10.0, 10.0));
        List<Double> res = Arrays.asList(10.0, 20.0, 20.0, 20.0);
        rt.train(lst, res);
        System.out.println(rt.predict(Arrays.asList(1.0)));
        rt.nodes.forEach(System.out::println);
    }


}
