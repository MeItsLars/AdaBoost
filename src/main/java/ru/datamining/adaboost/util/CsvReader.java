package ru.datamining.adaboost.util;

import lombok.SneakyThrows;

import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class CsvReader {

    @SneakyThrows
    public static List<List<String>> readCsvDataFile(File file) {
        return Files.readAllLines(file.toPath()).stream()
                .map(input -> Arrays.asList(input.split(";")))
                .collect(Collectors.toList());
    }

}
