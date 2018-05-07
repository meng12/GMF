package main;

import java.io.IOException;

import data_structure.DenseMatrix;
import utils.Printer;
import algorithms.MF_fastALS;
import algorithms.MF_ALS;
import algorithms.MF_weALS;
import algorithms.MF_CD;
import algorithms.ItemPopularity;
import data_structure.Rating;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;


public class main_MF extends main {
	public static void main(String argv[]) throws IOException {
		String dataset_name = "yelp";
		String method = "FastALS";
		double w0 = 10;
		boolean showProgress = false;
		boolean showLoss = true;
		int factors = 64;
		int maxIter = 500;
		double reg = 0.01;
                double lr = 0.01;
		double alpha = 0.75;
		
		if (argv.length > 0) {
			dataset_name = argv[0];
			method = argv[1];
			w0 = Double.parseDouble(argv[2]);
			showProgress = Boolean.parseBoolean(argv[3]);
			showLoss = Boolean.parseBoolean(argv[4]);
			factors = Integer.parseInt(argv[5]);
			maxIter = Integer.parseInt(argv[6]);
			reg = Double.parseDouble(argv[7]);
			if (argv.length > 8) alpha = Double.parseDouble(argv[8]);
                        if (argv.length > 9) lr = Double.parseDouble(argv[9]);
		}
		
               ReadRatings_HoldOneOut("data/" + dataset_name + ".rating_i10_u10");                		
	//	ReadRatings_GlobalSplit("data/" + dataset_name + ".rating_i10_u10", 0.1);
		
                System.out.printf("%s: showProgress=%s, factors=%d, maxIter=%d, reg=%f, w0=%.2f, alpha=%.2f, lr=%.3f\n",
				method, showProgress, factors, maxIter, reg, w0, alpha, lr);
		System.out.println("====================================================");
		
		ItemPopularity popularity = new ItemPopularity(trainMatrix, testRatings,  topK, threadNum);
		evaluate_model(popularity, "Popularity");
		
		double init_mean = 0;
		double init_stdev = 0.01;
                DenseMatrix U = new DenseMatrix(userCount, factors);
		DenseMatrix V = new DenseMatrix(itemCount, factors);
		U.init(init_mean, init_stdev, 111);
		V.init(init_mean, init_stdev, 1111);
		
		if (method.equalsIgnoreCase("fastals")) {
			MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings,  topK, threadNum,
					factors, maxIter, w0, alpha, reg, init_mean, init_stdev, showProgress, showLoss);
                        fals.setUV(U, V);
			evaluate_model(fals, "MF_fastALS");
                       // evaluate_model_online(fals, "MF_fastALS", 100);
		}
		
		if (method.equalsIgnoreCase("als")) {
			MF_ALS als = new MF_ALS(trainMatrix, testRatings,  topK, threadNum,
					factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss);
                        als.setUV(U, V);
			evaluate_model(als, "MF_ALS");
		}
                           		
                if (method.equalsIgnoreCase("weals")) {
			MF_weALS weals = new MF_weALS(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, reg, lr, init_mean, init_stdev, showProgress, showLoss);
                        weals.esetUV(U, V);
			evaluate_model(weals, "MF_weALS");  
                       // evaluate_model_online(weals, "MF_weALS", 100);
		}   
                                
		if (method.equalsIgnoreCase("cd")) {
			MF_CD cd = new MF_CD(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss);
			evaluate_model(cd, "MF_CD");
                //        evaluate_model_online(cd, "MF_cd", 100);
		}
		
		if (method.equalsIgnoreCase("all")) {
			
			MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings, topK, threadNum,
					factors, 200, 512, 0.4, reg, init_mean, init_stdev, showProgress, showLoss);
			fals.setUV(U, V);
			evaluate_model(fals, "MF_fastALS");
			
			MF_ALS als = new MF_ALS(trainMatrix, testRatings, topK, threadNum,
					factors, 200, 512, reg, init_mean, init_stdev, showProgress, showLoss);
			als.setUV(U, V);
			evaluate_model(als, "MF_ALS");
                        
                        
                        MF_weALS weals = new MF_weALS(trainMatrix, testRatings, topK, threadNum,
			factors, maxIter, 8, 0.3, 0.002, init_mean, init_stdev, showProgress, showLoss);
			weals.esetUV(U, V);
			evaluate_model(weals, "MF_weALS");
			
		//	MF_CD cd = new MF_CD(trainMatrix, testRatings, topK, threadNum,
		//			factors, maxIter, w0, reg, init_mean, init_stdev, showProgress, showLoss);
		//	cd.setUV(U, V);
		//	evaluate_model(cd, "MF_CD");
		}
                
                if (method.equalsIgnoreCase("eval_w0")) {			
			double[] array_w = {0.3,1.8,2,2.5,3.5};                      
                        // {128, 256, 512, 1024}; 
                        
			double[] res = new double[3];
                        PrintWriter writer1 = new PrintWriter (new FileOutputStream("eval_w0"));
                        
                        for (int s= 0; s< array_w.length; s++){
                        MF_weALS weals = new MF_weALS(trainMatrix, testRatings, topK, threadNum,
			factors, maxIter, 128, array_w[s], 0.002, init_mean, init_stdev, showProgress, showLoss);
			weals.esetUV(U, V);
			res = evaluate_model(weals, "MF_weALS");
                        
                    //    MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings, topK, threadNum,
			//		factors, maxIter, array_w[s], 0, 10, init_mean, init_stdev, showProgress, showLoss);
			//fals.setUV(U, V);
			//res = evaluate_model(fals, "MF_fastALS");
                        writer1.printf("%.5f\t%.5f\t%.5f\n", res[0], res[1], res[2]);
                        }
		       writer1.close();
		}
                
                if (method.equalsIgnoreCase("eval_alpha")) {	
			double[] array_alpha = {0.3, 0.4, 0.5, 0.6};
			double[] res = new double[3];
                        PrintWriter writer1 = new PrintWriter (new FileOutputStream("eval_alpha"));
                       
                        for (int s= 0; s< array_alpha.length; s++){
                      //  MF_weALS weals = new MF_weALS(trainMatrix, testRatings, topK, threadNum,
		 //	factors, maxIter, 8, array_alpha[s], reg, lr, init_mean, init_stdev, showProgress, showLoss);
		//	weals.esetUV(U, V);
		//	res = evaluate_model(weals, "MF_weALS");
                        
                         MF_fastALS fals = new MF_fastALS(trainMatrix, testRatings, topK, threadNum,
					factors, maxIter, 512, array_alpha[s], 1, init_mean, init_stdev, showProgress, showLoss);
			fals.setUV(U, V);
			res = evaluate_model(fals, "MF_fastALS");
                        writer1.printf("%.5f\t%.5f\t%.5f\n", res[0], res[1], res[2]);
                        }
		       writer1.close();
		}
	
	} // end main
}
