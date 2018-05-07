/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package algorithms;

import data_structure.Rating;
import data_structure.SparseMatrix;
import data_structure.DenseVector;
import data_structure.DenseMatrix;
import data_structure.Pair;
import data_structure.SparseVector;
import happy.coding.math.Randoms;
import happy.coding.math.Stats;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import utils.Printer;
/**
 * Algorithm for ALS under exponential family distribution
 * @author menglu
 */
public class MF_weALS extends TopKRecommender {
    	/** Model priors to set. */
	int factors = 10; 	// number of latent factors.
	int maxIter = 100; 	// maximum iterations.
	double w0 = 1;	
	double reg = 0.01; 	// regularization parameters
        double lr = 0.01;       //learning rate
        double init_mean = 0;  // Gaussian mean for init V
        double init_stdev = 0.01; // Gaussian std-dev for init V
               
	
        /** Model parameters to learn */
        public DenseMatrix U, oldU;	// latent vectors for users
        public DenseMatrix V, oldV;	// latent vectors for items 
        
        // Caches
        DenseMatrix SU, oldSU;
        DenseMatrix SV, oldSV;
        DenseMatrix Ugrad_se, oldUgrad_se;
        DenseMatrix Vgrad_se, oldVgrad_se;
        DenseMatrix deltaU_se;
        DenseMatrix deltaV_se;
        double[] prediction_users, prediction_items;
        double[] rating_users, rating_items;
        double[] w_users, w_items;
        double[] pre_uloss, pre_iloss, cur_uloss, cur_iloss, ulr, vlr;
 	
	boolean showProgress;
	boolean showLoss;
        public double[] hr_vec = new double[maxIter];
        public double[] ndcg_vec = new double[maxIter];
	public double[] precs_vec = new double[maxIter];
        
        // weight for each positive instance in trainMatrix
        SparseMatrix W;
        
        // weight for negative instances on item i.
        double[] Wi;
        // weight for users.
        double[] Au;
  
  // weight of new instance in online learning
  public double w_new = 1;
        double eloss = 0;
  
	public MF_weALS(SparseMatrix trainMatrix, ArrayList<Rating> testRatings, 
			int topK, int threadNum, int factors, int maxIter, double w0, double reg, double lr,
			double init_mean, double init_stdev, boolean showProgress, boolean showLoss) {
		super(trainMatrix, testRatings, topK, threadNum);
		this.factors = factors;
		this.maxIter = maxIter;
		this.w0 = w0;
		this.reg = reg;
                this.lr =lr;
		this.init_mean = init_mean;
		this.init_stdev = init_stdev;
		this.showProgress = showProgress;
		this.showLoss = showLoss;	               

         
		// assign weight
                double sum = 0, Z = 0;
		double[] p = new double[itemCount];
		for (int i = 0; i < itemCount; i ++) {
			p[i] = trainMatrix.getColRef(i).itemCount();
			sum += p[i];
		}
		// convert p[i] to probability 
		for (int i = 0; i < itemCount; i ++) {
			p[i] /= sum;
                        p[i] = Math.pow(p[i], 0);
			Z += p[i];
		}
		// assign weight
		Wi = new double[itemCount];
		for (int i = 0; i < itemCount; i ++)
			Wi[i] = w0* p[i]/Z;
                
                // discriminate selected new users
                Au = new double[userCount];
		for (int u = 0; u < userCount; u ++){
		   if (trainMatrix.getRowRef(u).itemCount() == 0)
                       Au[u] = 0;
                   else
                       Au[u] = 1;
                }
	

                
		// By default, the weight for positive instance is uniformly 1.
		W = new SparseMatrix(userCount, itemCount);
		for (int u = 0; u < userCount; u ++)
			for (int i : trainMatrix.getRowRef(u).indexList())
				W.setValue(u, i, 1);
                
                // Init caches
		prediction_users = new double[userCount];
		prediction_items = new double[itemCount];
		rating_users = new double[userCount];
		rating_items = new double[itemCount];
		w_users = new double[userCount];
		w_items = new double[itemCount];
                
                
                // Init model parameters
		U = new DenseMatrix(userCount, factors);
		V = new DenseMatrix(itemCount, factors);  
                               
		U.init(init_mean, init_stdev);
		V.init(init_mean, init_stdev);                                    
                initweS();
                
                Ugrad_se = new DenseMatrix(userCount, factors);
                Vgrad_se = new DenseMatrix(itemCount, factors);              
                
	}
        
        // Init SU and SV
	public void initweS() {		
                // Init SU as U^T Au U
		SU = new DenseMatrix(factors, factors);
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = 0;
				for (int u = 0; u < userCount; u ++) 
					val += U.get(u, f) * U.get(u, k) * Au[u];
				SU.set(f, k, val);
				SU.set(k, f, val);
			}
		}
		// Init SV as V^T Wi V
		SV = new DenseMatrix(factors, factors);
		for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = 0;
				for (int i = 0; i < itemCount; i ++) 
					val += V.get(i, f) * V.get(i, k) * Wi[i];
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		}
	}
        
        public void buildModel() throws IOException{
		     System.out.printf("Run for weALS: showProgress=%s, factors=%d, maxIter=%d, reg=%f, w0=%.2f, lr = %.3f",
				showProgress, factors, maxIter, reg, w0, lr);
		System.out.println("====================================================");
                PrintWriter writer = new PrintWriter (new FileOutputStream("MF_weALS.progress"));
		double[] res = new double[3];
		double loss_pre = Double.MAX_VALUE;
                double loss_cur;
                
                        
		for (int iter = 0; iter < maxIter; iter ++) {
		      Long start = System.currentTimeMillis();
                      System.out.printf("learning rate=%.5f\n", lr);
                                   
                     oldU = U;    oldV = V;
                     oldSU = SU;  oldSV = SV;
                     oldUgrad_se = Ugrad_se;  oldVgrad_se = Vgrad_se;
		      for (int u = 0; u < userCount; u ++) {
                                we_update_user(u);    
		      }                           
 
		      for (int i = 0; i < itemCount; i ++) {
				we_update_item(i);
                      }              
                        
			// Show progress
			if (showProgress)
                        {
				res = showProgress(iter, start, testRatings);
                                writer.printf("%.5f\t%.5f\t%.5f\n", res[0], res[1], res[2]);                 
                        }

                        // Show loss
                        if (showLoss){
                            loss_cur = show_eLoss(iter, start, loss_pre);
                         //   if(loss_cur > loss_pre){
                         //       U = oldU; V = oldV; SU= oldSU; SV = oldSV; 
                         //       Ugrad_se = oldUgrad_se; Vgrad_se = oldVgrad_se; 
                         //       lr = lr * 0.5;
                        //    }
                            if(loss_cur > loss_pre | (loss_pre -loss_cur) < 0.01) {                           
                              writer.close();
                              return;
                            }                          
                            else{
                            //  new_diff = loss_pre - loss_cur;
                           //   if (iter> 150 && new_diff > old_diff)  
                            //    lr = lr * 0.9;                                            
                            //  old_diff = new_diff;
                              loss_pre = loss_cur;
                                                         
                            // lr = lr * 1.05;
                            //  lr = pre_lr/ Math.sqrt(iter + 1); // pre_lr = 0.05
                            //  lr = pre_lr/Math.exp((iter+1)*0.02);  //conscequence: decrease not smoothly 
                           //   lr = pre_lr/ (1+ iter * 0.1);
                           // if(iter!=0 && iter%10 == 0)  lr = lr * 0.6;   // decay after constant steps
                            } 
                         
                         // Adjust by ndcg
                            //    evaluate(testRatings);
			//	double hr = ndcgs.mean();			
                            //    if (hr > hr_prev)
                            //        lr = lr *1.05;
                           //     else{
                          //          U = oldU; V = oldV; SU= oldSU; SV = oldSV; 
                          //      Ugrad_se = oldUgrad_se; Vgrad_se = oldVgrad_se; 
                          //      lr = lr * 0.5;
                         //       }
                         //       hr_prev = hr;   
                                
                        }                   
			
		}
                writer.close();
       }
       
        
       	//remove
	public void esetUV(DenseMatrix U, DenseMatrix V) {
		this.U = U.clone();
		this.V = V.clone();
		initweS();
	}
        
        
       public double show_eLoss(int iter, long start, double loss_pre){
		long start1 = System.currentTimeMillis();
		double loss_cur = e_loss();
		String symbol = loss_pre >= loss_cur ? "-" : "+";
		System.out.printf("Iter=%d [%s]\t [%s]loss: %.4f diff: %.4f [%s]\n", iter, 
				Printer.printTime(start1 - start), symbol, loss_cur, loss_pre-loss_cur, 
				Printer.printTime(System.currentTimeMillis() - start1));                
		return loss_cur;
	}
       

       public double e_loss() {
                double L = reg/2 * (U.squaredSum() + V.squaredSum());
		for (int u = 0; u < userCount; u ++) {                   
                        double l = 0;
			for (int i : trainMatrix.getRowRef(u).indexList()) {
			    double cta = predict(u,i);                                           
                            l += W.getValue(u, i) *(Math.log(1+ Math.exp(cta)) - trainMatrix.getValue(u, i) * cta);                        
                            l -= 1/2* Wi[i] * Math.pow(cta, 2) ;  
			}
                        l += 1/2 * SV.mult(U.row(u, false)).inner(U.row(u, false));
			L += l;                    
		}	
		return L;
       }

       
       private void we_update_user(int u){               
           ArrayList<Integer> itemList = trainMatrix.getRowRef(u).indexList();
           if (itemList.size() == 0)	return;
           
                // prediction cache for the user
		for (int i : itemList) {
			prediction_items[i] = predict(u, i);
			rating_items[i] = trainMatrix.getValue(u, i);
			w_items[i] = W.getValue(u, i);                        
		}
               
                DenseVector oldVector = U.row(u);
		for (int f = 0; f < factors; f ++) {
			double neg = 0, pos = 0, grad = 0;
			// O(K) complexity for the negative part
			for (int k = 0; k < factors; k ++) {
				
					neg += U.get(u, k) * SV.get(f, k);
			}
						
			// O(Nu) complexity for the positive part
			for (int i : itemList) {
                            double tt = 0;
                            tt += w_items[i]* (drvA(prediction_items[i])- rating_items[i]);
                            tt -= Wi[i]* prediction_items[i];
                            pos += tt * V.get(i,f);
                        }
                        grad = neg + pos + reg * U.get(u,f);
                       
                       Ugrad_se.set(u, f,  0.9 * Ugrad_se.get(u,f) + 0.1 * Math.pow(grad,2));  
                       // Parameter Update
		       U.set(u, f, U.get(u, f) - lr * grad/Math.sqrt(Ugrad_se.get(u,f)+ 1e-10));                                                                           
                }
                
                // Calculate log-likelihood l for user u;
        //       cur_uloss[u] = u_loss(u);
       //        
      //         if(cur_uloss[u] > pre_uloss[u])  { // roll back
        //           U.setRow(u, oldVector); 
         ///          Ugrad_se.setRow(u, oldUgradseVec); 
         //          ulr[u] = ulr[u] * 0.5;
          //     }                                     
          //     else{                            
                   // Update the SU cache
		   for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SU.get(f, k) - oldVector.get(f) * oldVector.get(k)
						+ U.get(u, f) * U.get(u, k);
				SU.set(f, k, val);
				SU.set(k, f, val);
			}		   
                   } // end for f
                   
             //      pre_uloss[u] = cur_uloss[u];
            //       ulr[u] = ulr[u] * 1.05;
             //  }    
       }   
           
     
              
       private double drvA (double x){
           return 1- 1/(1+ Math.exp(x));
       }           

                     
       
       private void we_update_item(int i){
           ArrayList<Integer> userList = trainMatrix.getColRef(i).indexList();
           
                if (userList.size() == 0)	return;
           
                // prediction cache for the item
		for (int u : userList) {
			prediction_users[u] = predict(u, i);
			rating_users[u] = trainMatrix.getValue(u, i);
			w_users[u] = W.getValue(u, i);                        
		}
           
                DenseVector oldVector = V.row(i);
		for (int f = 0; f < factors; f ++) {
			double neg = 0, pos = 0, grad = 0;
			// O(K) complexity for the negative part
			for (int k = 0; k < factors; k ++) {
				
					neg += Wi[i] * V.get(i, k) * SU.get(f, k);
			}
			
			
			// O(Nu) complexity for the positive part
			for (int u : userList) {
                            double tt = 0;
                            tt += w_users[u]* (drvA(prediction_users[u])- rating_users[u]);
                            tt -= Wi[i]* prediction_users[u];
                            pos += tt * U.get(u,f);
                        }
                        grad = neg + pos + reg * V.get(i,f);
                     
                       Vgrad_se.set(i, f,  0.9 * Vgrad_se.get(i,f) + 0.1 * Math.pow(grad,2));                       
                           // Parameter Update
		       V.set(i, f, V.get(i, f) - lr * grad/Math.sqrt(Vgrad_se.get(i,f)+ 1e-10));                                             
                }
                
               // Calculate log-likelihood l for item i;
               //cur_iloss[i] = i_loss(i);
               
              // if(cur_iloss[i] > pre_iloss[i])  { // roll back
             //      V.setRow(i, oldVector); 
             //      Vgrad_se.setRow(i, oldVgradseVec); 
             //      eloss += pre_iloss[i];
             //      vlr[i] = vlr[i] * 0.5;
             //  }                                     
             //  else{                            
                    // Update the SV cache
		    for (int f = 0; f < factors; f ++) {
			for (int k = 0; k <= f; k ++) {
				double val = SV.get(f, k) - oldVector.get(f) * oldVector.get(k)* Wi[i]
						+ V.get(i, f) * V.get(i, k) * Wi[i];
				SV.set(f, k, val);
				SV.set(k, f, val);
			}
		    } // end for f
                   
              //     eloss += cur_iloss[i];
              //     pre_iloss[i] = cur_iloss[i];
              //     vlr[i] = vlr[i] * 1.05;
             //  }    
               
       }  

    
       @Override
	public double predict(int u, int i) {
		return U.row(u, false).inner(V.row(i, false));
	}
        
        
        @Override 
	public void updateModel(int u, int i) {
		trainMatrix.setValue(u, i, 1);
		W.setValue(u, i, w_new);
                
		if (Wi[i] == 0) { // a new item
			Wi[i] = w0 / itemCount;
			// Update the SV cache
			for (int f = 0; f < factors; f ++) {
				for (int k = 0; k <= f; k ++) {
					double val = SV.get(f, k) + V.get(i, f) * V.get(i, k) * Wi[i];
					SV.set(f, k, val);
					SV.set(k, f, val);
				}
			}
		}
                
                if (Au[u] == 0) { // a new user		
                        Au[u] = 1;
			// Update the SU cache
                        for (int f = 0; f < factors; f ++) {
			   for (int k = 0; k <= f; k ++) {				
				double val = SU.get(f,k) + U.get(u, f) * U.get(u, k);
				SU.set(f, k, val);
				SU.set(k, f, val);
			   }
		        }			
		}
                
                double online_loss_pre = e_loss();  
                
		for (int iter = 0; iter < maxIterOnline; iter ++) {                      
			we_update_user(u);			
			we_update_item(i);
                        double online_loss_cur = e_loss();
                        
                        if(online_loss_cur > online_loss_pre | (online_loss_pre -online_loss_cur) < 0.001)                                                    
                           {System.out.printf("iter%d diff:%.4f\n", iter+1, online_loss_pre -online_loss_cur);  return; }                                                     
                  //      else
                    //         System.out.printf("%.4f ", online_loss_pre -online_loss_cur);
                         if(iter == maxIterOnline-1)
                            System.out.printf("iter%d diff:%.4f\n", iter+1, online_loss_pre -online_loss_cur); 
                         
                         online_loss_pre = online_loss_cur;
		}
	}
}

