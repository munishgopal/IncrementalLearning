package edu.iastate.cs;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import edu.iastate.cs.Utilities;
import weka.core.Instances;

public class NBayes {
	
	
	public static void main(String args[]) throws Exception{
		
		Instances D1;
		Instances D2;
		
		Utilities u = new Utilities();
		
		BufferedReader inputReader = new BufferedReader(new FileReader("TrainingFileLocation.txt"));
		BufferedReader outputReader = new BufferedReader(new FileReader("TestFileLocation.txt"));
			
		FileWriter fw = new FileWriter("samp.xls");
		String inputLine = inputReader.readLine();
		String outputLine = outputReader.readLine();
		
		while(inputLine!=null){
			System.out.println(inputLine);
			Instances train = u.readData(inputLine);
			Instances test = u.readData(outputLine);
			
			System.out.println(train.numInstances());
			fw.write("\n\n");
			for (double i= 0.5; i <1 ; i = i+0.5) {
			int count = (int) Math.ceil((double)train.numInstances() * i);
			System.out.println(count);
			
			D1 = new Instances(train,0,count);
			D2 = new Instances(train, count, train.numInstances() - count);
			
			D1.setClassIndex(train.numAttributes() - 1);
			D2.setClassIndex(train.numAttributes() - 1);
			
			//Adding Data Test.
			System.out.println("Whole Model:");
			long startTime = System.currentTimeMillis();
			NB n = new NB();
			n.buildModel(train);
			n.classifyAndEvaluate(test);
			long endTime = System.currentTimeMillis();
			System.out.println("Time Taken :" +(endTime-startTime));
			fw.write(endTime-startTime+"\t");
			
			FileOutputStream fos = new FileOutputStream(inputLine+"1");
			FileInputStream fis = new FileInputStream(inputLine+"1");
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			ObjectInputStream ios = new ObjectInputStream(fis);
			
			
			NB nb = new NB();
			nb.buildModel(D1);
			oos.writeObject(nb);
			oos.close();
			System.out.println("Incremental Model:");
			startTime = System.currentTimeMillis();
			nb.addData(D2);
			nb.classifyAndEvaluate(test);
			
			endTime = System.currentTimeMillis();
			System.out.println("Time Taken :" + (endTime-startTime));
			fw.write(endTime-startTime+"\t");
			
			System.out.println();
			//Remove Data Test.
			System.out.println("Remove Data");
			System.out.println("Whole Model:");
			startTime = System.currentTimeMillis();
			NB nRemove = new NB();
			nRemove.buildModel(D2);
			nRemove.classifyAndEvaluate(test);
			endTime = System.currentTimeMillis();
			System.out.println("Time Taken :" +(endTime-startTime));
			fw.write(endTime-startTime+"\t");
			
			System.out.println("Incremental Model:");
			startTime = System.currentTimeMillis();
			nb.removeData(D1);
			nb.classifyAndEvaluate(test);
			endTime = System.currentTimeMillis();
			System.out.println("Time Taken :" + (endTime-startTime));
			fw.write(endTime-startTime+"\t");
			fw.write("\n");
		}
			inputLine = inputReader.readLine();
			outputLine = outputReader.readLine();
		}
		fw.close();
	}
}

class NB implements Serializable {
	
	HashMap <String,double[]> attributeCountsForClass[];
	
	double p_class1;
	double p_class2;
	
	String[] attributeValue;
	
	public void removeData(Instances D) {
		for(int j=0;j<D.numInstances();j++){
			String attr[] = D.instance(j).toString().split(",");
			
			int classvalue = Integer.parseInt(D.instance(j).toString().split(",")[attr.length-1]);
			for(int c =0;c<attr.length-1;c++){ 
			    attributeCountsForClass[classvalue].get(attr[c])[0] = attributeCountsForClass[classvalue].get(attr[c])[0]-1;
			}
			//System.gc();
		}
		calculateProbabilities(D);
	}
	
	public void addData(Instances D) {
		
		for(int j=0;j<D.numInstances();j++){
			String attr[] = D.instance(j).toString().split(",");
			
			int classvalue = Integer.parseInt(D.instance(j).toString().split(",")[attr.length-1]);
			for(int c =0;c<attr.length-1;c++){ 
			    attributeCountsForClass[classvalue].get(attr[c])[0] = attributeCountsForClass[classvalue].get(attr[c])[0]+1;
			}
			//System.gc();
		}
		calculateProbabilities(D);
	}

	public void buildModel(Instances D) {
		initialise(D);
		addData(D);
	}
	
	private void initialise(Instances D) {
		
		String attributeValues = D.attribute(0).toString();
		attributeValues = attributeValues.substring(attributeValues.indexOf("{")+1,attributeValues.lastIndexOf("}"));
		attributeValue = attributeValues.split(",");
		attributeCountsForClass = new HashMap[2];
		attributeCountsForClass[0] = new HashMap<String,double[]>();
		attributeCountsForClass[1] = new HashMap<String,double[]>();	
				
		for(int i=0;i<attributeValue.length;i++){
			double [] temp1 = {0.0,0.0};
			double [] temp2 = {0.0,0.0};
			attributeCountsForClass[0].put(attributeValue[i], temp1);
			attributeCountsForClass[1].put(attributeValue[i], temp2);
		}
		
	}

	public void classifyAndEvaluate(Instances test) {
		double correct = 0;
		double incorrect = 0;
		
		for(int i=0;i<test.numInstances();i++) {
			String attributeValues[] = test.instance(i).toString().split(",");
			double prob1 = p_class1;
			double prob2 = p_class2;
			for(int c=0;c<attributeValues.length-1;c++) {
				double [] val1 = attributeCountsForClass[0].get(test.instance(i).toString().split(",")[c]);
				double[] val2 = attributeCountsForClass[1].get(test.instance(i).toString().split(",")[0]);
					
				prob1 *= val1[1];
				prob2 *= val2[1];
			}
			
			//System.out.println(Integer.parseInt(test.instance(i).toString().split(",")[1]));
			if((prob1 > prob2) && (Integer.parseInt(test.instance(i).toString().split(",")[attributeValues.length-1])  == 0)) {
				//System.out.println(test.instance(i).toString().split(",")[0] + "---" + class1 +"---" + class2);
				correct++;
			}
			else incorrect++;
		}
		double answer = correct / (double)(test.numInstances()) * 100;
		System.out.println("Correctly Classified:" + answer);
	}
	
	private void calculateProbabilities(Instances D) {
		
		int [] attributeCountsForAllClass = D.attributeStats(0).nominalCounts;
		
		Set keys1 = attributeCountsForClass[0].keySet();
		Set keys2 = attributeCountsForClass[1].keySet();
	
		Iterator i = keys1.iterator();
		Iterator j = keys2.iterator();
		
		int c =0;
		
		double totalcount = 0;
		double class1count = 0;
		double class2count = 0;
		
		while(i.hasNext()){
			String k = (String) i.next();
			String k1 = (String) j.next();
			double[] vals = attributeCountsForClass[0].get(k);
			class1count += vals[0];
			double[] vals1 = attributeCountsForClass[1].get(k1);
			class2count += vals1[0];
		}
		
		totalcount = class1count + class2count;
		
		i = keys1.iterator();
		j = keys2.iterator();
		
		while(i.hasNext()){
			String k = (String) i.next();
			String k1 = (String) j.next();
			
			double [] vals;
			
			vals = attributeCountsForClass[0].get(k);
			vals[1] = vals[0]  / class1count;
			attributeCountsForClass[0].put(k,vals);
			
			vals = attributeCountsForClass[1].get(k1);
			vals[1] = vals[0] / class2count;
			attributeCountsForClass[1].put(k1,vals);
		}

		p_class1 = class1count / totalcount ;
		p_class2 = class2count / totalcount ;
 	
	}
}
