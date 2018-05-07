package main;

import java.io.IOException;

import utils.Printer;
import algorithms.MFbpr;
import algorithms.ItemPopularity;
import algorithms.TopKRecommender;
import data_structure.DenseMatrix;
import data_structure.Rating;

import java.util.ArrayList;
import static main.main.itemCount;
import static main.main.userCount;

public class main_bpr extends main {
	public static void main(String argv[]) throws IOException {
		String dataset_name = "yelp";
		int factors = 64 ;
		double lr = 0.04;
		double reg = 0.01;
		int num_dns = 1; // number of dynamic negative samples [Zhang Weinan et al. SIGIR 2013]
		int maxIter = 1000;
		double init_mean = 0;
		double init_stdev = 0.01;
		
		if (argv.length > 0) {
			dataset_name = argv[0];
			factors = Integer.parseInt(argv[1]);
			lr = Double.parseDouble(argv[2]);
			reg = Double.parseDouble(argv[3]);
		}
		ReadRatings_HoldOneOut("data/" + dataset_name + ".rating_i10_u10");
               // ReadRatings_GlobalSplit("data/" + dataset_name + ".rating_i10_u10", 0.1);
		topK = 100;
		
		System.out.printf("BPR with factors=%d, lr=%.4f, reg=%.4f, num_dns=%d\n", 
				factors, lr, reg, num_dns);
		System.out.println("====================================================");
		
		ItemPopularity pop = new ItemPopularity(trainMatrix, testRatings,  topK, threadNum);
		evaluate_model(pop, "Popularity");
                
                DenseMatrix U = new DenseMatrix(userCount, factors);
		DenseMatrix V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev, 111);
		V.init(init_mean, init_stdev, 1111);
		
		MFbpr bpr = new MFbpr(trainMatrix, testRatings, topK, threadNum, 
				factors, maxIter, lr, false, reg, init_mean, init_stdev, num_dns, true);
                bpr.bsetUV(U,V);
		evaluate_model(bpr, "BPR");
	//	evaluate_model_online(bpr, "BPR", 100);
	} // end main
}
