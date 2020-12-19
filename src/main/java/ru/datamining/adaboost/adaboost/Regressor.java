package ru.datamining.adaboost.adaboost;

import java.util.List;

/**
 * Interface for representing a regressor.
 */
public interface Regressor {

    /**
     * Trains the regressor on the given input data, with the given results
     * The size of the 'data' list should be equal to the size of the 'expectedResults' list
     *
     * @param data            The input data, formatted into a list of data entries
     * @param expectedResults The expected results, formatted into a list of values
     */
    void train(List<List<Double>> data, List<Double> expectedResults);

    /**
     * Predicts the output value of the given input entry
     *
     * @param entry The input entry
     * @return The predicted output
     */
    double predict(List<Double> entry);

}