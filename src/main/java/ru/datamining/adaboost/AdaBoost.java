package ru.datamining.adaboost;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class AdaBoost {

    public static void main(String[] args) throws IOException {
        File file = new File("./data/student.txt");

        System.out.println(Files.readAllLines(file.toPath()));
    }

}
