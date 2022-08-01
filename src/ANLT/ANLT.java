package ANLT;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;

import LFTcommom.InitTensor;
import LFTcommom.TensorTuple;


public class ANLT  extends InitTensor{

	ANLT(String trainFile, String testFile, String separator )
	{
		super(trainFile, testFile, separator); 
	}
	 
	public void train() throws IOException 
	{
		long startTime = System.currentTimeMillis();   
		
		FileWriter  fw = new FileWriter(new File(trainFile.replace(".txt", "_")+rank+"_"+eta+"_"
				+lambda+"_"+new Date().getTime() / 1000+"_ANLT.txt"));
		fw.write("round :: everyRoundRMSE :: everyRoundMAE :: costTime(ms) \n");
		fw.flush();
			
		initFactorMatrix();
		initAssistMatrix();
		initSliceSet();
		initLagAssit(); 
		logNormalization();   // data-dependent
		
		for(TensorTuple trainTuple: trainData)
		{
			trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
		}
		
		for(int round = 1; round <= trainRound; round++)
		{
			long startRoundTime = System.currentTimeMillis();    
			initAssistMatrix();
			
			double[] Temp1 = new double[maxAID+1];
			double[] Temp2 = new double[maxBID+1];
			double[] Temp3 = new double[maxCID+1];
			
			for(int r = 1; r <= rank; r++)
			{
				for (int max_a_id = 1; max_a_id <= maxAID; max_a_id++) 
				{
					Temp1[max_a_id] = Sp[max_a_id][r];
				}

				for (int max_b_id = 1; max_b_id <= maxBID; max_b_id++) 
				{
					Temp2[max_b_id] = Dp[max_b_id][r];
				}

				for (int max_c_id = 1; max_c_id <= maxCID; max_c_id++) 
				{
					Temp3[max_c_id] = Tp[max_c_id][r];
				}
				
				
				for(TensorTuple trainTuple: trainData)
				{
					double error = trainTuple.value - trainTuple.valueHat + Sp[trainTuple.aID][r] *
							Dp[trainTuple.bID][r] * Tp[trainTuple.cID][r];
									
					Spup[trainTuple.aID][r] += error * Dp[trainTuple.bID][r] * Tp[trainTuple.cID][r];
					Spdown[trainTuple.aID][r] += Math.pow(Dp[trainTuple.bID][r] * Tp[trainTuple.cID][r], 2);
					
					Dpup[trainTuple.bID][r] += error * Sp[trainTuple.aID][r] * Tp[trainTuple.cID][r];
					Dpdown[trainTuple.bID][r] += Math.pow(Sp[trainTuple.aID][r] * Tp[trainTuple.cID][r], 2);
					
					Tpup[trainTuple.cID][r] += error * Sp[trainTuple.aID][r] * Dp[trainTuple.bID][r];
					Tpdown[trainTuple.cID][r] += Math.pow(Sp[trainTuple.aID][r] * Dp[trainTuple.bID][r], 2);
				}
				
				for(int i = 1; i <= this.maxAID; i++)
				{
					Spup[i][r] += (alpha[i] * S[i][r] - M[i][r]);
					Spdown[i][r] += alpha[i];
					
					Sp[i][r] = Spup[i][r] / Spdown[i][r];
					
					double temps = Sp[i][r] + M[i][r] / alpha[i];
					if(temps >= 0)
					{
						S[i][r] = temps;
					}
					else
					{
						S[i][r] = 0;
					}
					
					M[i][r] += eta * alpha[i] * (Sp[i][r] - S[i][r]);  
				}

				for(int j = 1; j <= this.maxBID; j++)
				{
					Dpup[j][r] += (beta[j] * D[j][r] - N[j][r]);
					Dpdown[j][r] += beta[j];
					
					Dp[j][r] = Dpup[j][r] / Dpdown[j][r];
					
					double tempd = Dp[j][r] + N[j][r] / beta[j];
					if(tempd >= 0)
					{
						D[j][r] = tempd;
					}
					else
					{
						D[j][r] = 0;
					}
					
					N[j][r] += eta * beta[j] * (Dp[j][r] - D[j][r]);  
				}
	
				for(int k = 1; k <= this.maxCID; k++)
				{
					Tpup[k][r] += (gamma[k] * T[k][r] - Z[k][r]);
					Tpdown[k][r] += gamma[k];
					
					Tp[k][r] = Tpup[k][r] / Tpdown[k][r];
					
					double tempt = Tp[k][r] + Z[k][r] / gamma[k];
					if(tempt >= 0)
					{
						T[k][r] = tempt;
					}
					else
					{
						T[k][r] = 0;
					}
					
					Z[k][r] += eta * gamma[k] * (Tp[k][r] - T[k][r]); 
				}
				
				for(TensorTuple trainTuple: trainData)
				{
					trainTuple.valueHat = trainTuple.valueHat + Sp[trainTuple.aID][r] * Dp[trainTuple.bID][r]
							* Tp[trainTuple.cID][r] - Temp1[trainTuple.aID] * Temp2[trainTuple.bID]
									* Temp3[trainTuple.cID];
				}
				
			}
			
			
			for (int max_a_id = 1; max_a_id <= maxAID; max_a_id++) 
			{
				Temp1[max_a_id] = ap[max_a_id];
			}

			for (int max_b_id = 1; max_b_id <= maxBID; max_b_id++) 
			{
				Temp2[max_b_id] = bp[max_b_id];
			}

			for (int max_c_id = 1; max_c_id <= maxCID; max_c_id++) 
			{
				Temp3[max_c_id] = cp[max_c_id];
			}
			
			
			for(TensorTuple trainTuple: trainData)
			{			
				double error =  trainTuple.value - trainTuple.valueHat + ap[trainTuple.aID];

				apup[trainTuple.aID] += error;
			}
		
			for(int i = 1; i <= this.maxAID; i++)
			{
				apup[i] += (rho[i] * a[i] - p[i]);
				apdown[i] = aSliceSet[i] + rho[i];  
				
				ap[i] = apup[i] / apdown[i];
				
				double tempa = ap[i] + p[i] / rho[i];
				if(tempa >=0)
				{
					a[i] = tempa;
				}
				else
				{
					a[i] = 0;
				}

				p[i] += eta * rho[i] * (ap[i] - a[i]);
			}
			
			
			for(TensorTuple trainTuple: trainData)
			{

				double error =  trainTuple.value - trainTuple.valueHat + bp[trainTuple.bID];
				bpup[trainTuple.bID] += error;
			}
		
			for(int j = 1; j <= this.maxBID; j++)
			{
				bpup[j] += (nu[j] * b[j] - q[j]);
				bpdown[j] = bSliceSet[j] + nu[j];
				
				bp[j] = bpup[j] / bpdown[j];
				
				double tempb = bp[j] + q[j] / nu[j];
				if(tempb >=0)
				{
					b[j] = tempb;
				}
				else
				{
					b[j] = 0;
				}
				
				q[j] += eta * nu[j] * (bp[j] - b[j]);
			}

			
			for(TensorTuple trainTuple: trainData)
			{
				double error =  trainTuple.value - trainTuple.valueHat + cp[trainTuple.cID];		
				cpup[trainTuple.cID] += error;
			}
		
			for(int k = 1; k <= this.maxCID; k++)
			{
				cpup[k] += (delta[k] * c[k] - h[k]);
				cpdown[k] = cSliceSet[k] + delta[k];
				
				cp[k] = cpup[k] / cpdown[k];
				
				double tempc = cp[k] + h[k] / delta[k];
				if(tempc >=0)
				{
					c[k] = tempc;
				}
				else
				{
					c[k] = 0;
				}
				
				h[k] += eta * delta[k] * (cp[k] - c[k]);
			}
			
			
			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = trainTuple.valueHat + ap[trainTuple.aID] + bp[trainTuple.bID]
						+ cp[trainTuple.cID] - Temp1[trainTuple.aID]- Temp2[trainTuple.bID]
								- Temp3[trainTuple.cID];
			}
			

			double square = 0, absCount = 0;
			for (TensorTuple testTuple : testData) {
				testTuple.valueHat = this.getPrediction(testTuple.aID, testTuple.bID, testTuple.cID);
				square += Math.pow(testTuple.value - testTuple.valueHat, 2);
				absCount += Math.abs(testTuple.value - testTuple.valueHat);
			}
					
			everyRoundRMSE[round] = Math.sqrt(square / testCount);
			everyRoundMAE[round] = absCount / testCount; 
			
			long endRoundTime = System.currentTimeMillis();
			sumTime += (endRoundTime-startRoundTime);
			
			System.out.println(round + "::" + everyRoundRMSE[round] + "::" + everyRoundMAE[round]
					+"::"+ (endRoundTime-startRoundTime));
			
			fw.write(round + "::" + everyRoundRMSE[round] + "::" + everyRoundMAE[round]
					+"::"+ (endRoundTime-startRoundTime)+"\n");
			fw.flush();
			
	
			if (everyRoundRMSE[round-1] - everyRoundRMSE[round] > errorgap) 
			{
				if(minRMSE > everyRoundRMSE[round])
				{
					minRMSE = everyRoundRMSE[round];
					minRMSERound = round;
				}

				flagRMSE = false;
				tr = 0;
			}
			
			if (everyRoundMAE[round-1] - everyRoundMAE[round] > errorgap) 
			{
				if(minMAE > everyRoundMAE[round])
				{
					minMAE = everyRoundMAE[round];
					minMAERound = round;
				}

				flagMAE = false;
				tr = 0;
			} 
		
			if(flagRMSE && flagMAE)
			{
				tr++;
				if(tr == threshold)                
					break;
			}
			
			flagRMSE = true;
			flagMAE = true;
			
		}
		
		long endTime = System.currentTimeMillis();

		fw.write("Total Time Cost："+(endTime-startTime)/1000+"s\n");
		fw.flush();
		fw.write("minRMSE:"+minRMSE+"  minRSMERound "+minRMSERound+"\n");
		fw.flush();
		fw.write("minMAE:"+minMAE+"  minMAERound "+minMAERound+"\n");
		fw.flush();
		fw.write("rank="+rank+"\n");
		fw.flush();
		fw.write("trainCount: "+trainCount+"   testCount: "+testCount+"\n");
		fw.flush();
		fw.close();

		System.out.println("***********************************************");
		System.out.println("rank: "+this.rank+"\n");
		System.out.println("minRMSE:"+minRMSE+"  minRSMERound"+minRMSERound);
		System.out.println("minMAE:"+minMAE+"  minMAERound"+minMAERound);
		System.out.println("Total Time Cost："+(endTime-startTime)/1000+"s\n");	

	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		ANLT anlt = new ANLT("./result/trainingData.txt",
				"./result/testingData.txt", "::");
		
		anlt.threshold = 50;
		anlt.rank = 20;
		anlt.trainRound = 500;
		anlt.errorgap = 1E-6;
		
		anlt.eta = 1;   //data-dependent
		anlt.lambda = 2; //data-dependent 

		
		try {
			anlt.initData(anlt.trainFile, anlt.trainData, 0);
			anlt.initData(anlt.testFile, anlt.testData, 1);
			anlt.train();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}
