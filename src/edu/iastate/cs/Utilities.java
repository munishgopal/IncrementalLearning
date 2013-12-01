package edu.iastate.cs;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.Instances;

public class Utilities {
	public Instances readData(String filePath) throws Exception{
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		Instances inputData = new Instances(reader);
		reader.close();
		return inputData;
	}
}
