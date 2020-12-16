package ru.datamining.adaboost;

import ru.datamining.adaboost.util.CsvReader;

import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        File file = new File("./data/student-por.csv");

        System.out.println(CsvReader.readCsvDataFile(file).get(1));
    }

}