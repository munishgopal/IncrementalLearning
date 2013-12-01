package edu.iastate.cs;

import weka.core.Instance;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

public class BasicTest {
	public static void main(String[] args) throws Exception {
		BufferedReader inputReader = new BufferedReader(new FileReader("TrainingFileLocation.txt"));
		BufferedReader outputReader = new BufferedReader(new FileReader("TestFileLocation.txt"));
		
		String inputLine = inputReader.readLine();
		String outputLine = outputReader.readLine();
		
		while(inputLine!=null){
			
			Instances train = readData(inputLine);
			Instances test = readData(outputLine);
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(train.numAttributes() - 1);
			
			inputLine = inputReader.readLine();
			NaiveBayes NB = new NaiveBayes();
			long startTime = System.currentTimeMillis();
			NB.buildClassifier(train);
			
			double correct = 0;
			double incorrect = 0;
			
			for(int i=0;i<test.numInstances();i++) {
				double clas = NB.classifyInstance(test.instance(i));
				if((clas  == test.instance(i).classValue())) {
					correct++;
				}
				else incorrect++;
			}
			
			double answer = correct / (double)(test.numInstances()) * 100;
			System.out.println("Correctly Classified:" + answer);
			System.out.println("InCorrectly Classified:" + ((incorrect/test.numInstances()) * 100));
			long endTime = System.currentTimeMillis();
			System.out.println("Time Taken :" +(endTime-startTime));
			
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(NB, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			break;
		}
	}
	
	public static Instances readData(String filePath) throws Exception{
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		Instances inputData = new Instances(reader);
		reader.close();
		return inputData;
	}
}
