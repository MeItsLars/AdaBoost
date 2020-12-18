package ru.datamining.adaboost.adaboost;

import java.util.*;

public class RegressionTree implements Predictor {

    private final int minNrObservations;
    private final int maxDept;
    private BinaryNode rootNode;

    List<BinaryNode> nodes = new LinkedList<>();

    public RegressionTree(int maxDept, int minNrObservations) {
        this.maxDept = maxDept;
        this.minNrObservations = minNrObservations;
    }

    public void train(List<List<Double>> data, List<Double> results) {
        this.rootNode = new BinaryNode();
        train(data, results, rootNode);
    }


    public void train(List<List<Double>> data, List<Double> results, BinaryNode node) {
        System.out.println("Training a node at level " + node.getLevel());
        node.setAverage(results.stream().mapToDouble(v -> v).average().orElse(0));

        if (node.getLevel() >= maxDept) {
            System.out.println("Created a leaf node at level " + node.getLevel());
            return;
        }

        node.computeBestSplit(data, results);

        if (node.getSumSquaredError() == 0) {
            return;
        }

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

        BinaryNode leftNode = new BinaryNode(node);
        BinaryNode rightNode = new BinaryNode(node);

        node.setLeftChild(leftNode);
        node.setRightChild(rightNode);

        nodes.add(leftNode);
        nodes.add(rightNode);
        nodes.add(node);

        System.out.println("Left size: " + leftNodeData.size() + ", right size: " + rightNodeData.size());
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


    public static void main(String[] args) {
        RegressionTree rt = new RegressionTree(5, 1);
        List<List<Double>> lst = new ArrayList<>();
        /*lst.add(Arrays.asList(1.0, 3.0, 4.0));
        lst.add(Arrays.asList(5.0, 6.0, 7.0));
        lst.add(Arrays.asList(8.0, 9.0, 10.0));
        lst.add(Arrays.asList(10.0, 10.0, 10.0));
        List<Double> res = Arrays.asList(10.0, 20.0, 20.0, 20.0);*/
        lst.add(Collections.singletonList(1.0));
        lst.add(Collections.singletonList(2.0));
        lst.add(Collections.singletonList(3.0));
        lst.add(Collections.singletonList(4.0));
        rt.train(lst, Arrays.asList(1.0, 2.0, 3.0, 4.0));
        System.out.println(rt.predict(Arrays.asList(5.0)));
        //rt.nodes.forEach(System.out::println);
    }


}
