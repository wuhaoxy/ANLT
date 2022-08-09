package LFTcommom;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;


 
public class InitTensor {

	public int rank;            // tensor rank
	public int trainRound = 1000;    //Iteration Count
	public int minRMSERound = 0;     
	public int minMAERound = 0;          
	public double minRMSE = 100; 
	public double minMAE = 100; 
	public double everyRoundRMSE[], everyRoundMAE[]; 
	public double errorgap = 1E-1;      // error difference 
	public int threshold = 0;  
	public int tr = 0; 
	public boolean flagRMSE = true, flagMAE = true; 
	
	public ArrayList<TensorTuple> trainData = new ArrayList<TensorTuple>();  //training data
	public ArrayList<TensorTuple> testData = new ArrayList<TensorTuple>();   //validation data or testing data
	public int trainCount = 0;   // training entry count
	public int testCount = 0;   //validation/testing entry count
	public int maxAID = 0, maxBID = 0, maxCID = 0;
	public int minAID = Integer.MAX_VALUE, minBID = Integer.MAX_VALUE, minCID = Integer.MAX_VALUE; 

	public double maxValue = -1; 
	public double minValue = Integer.MAX_VALUE; 
	
	public double[] aSliceSet, bSliceSet, cSliceSet; 
	
	public double[][] S, D, T, Sp, Dp, Tp; // latent features matrices and auxiliary matrices
	public double[][] Spup, Spdown, Dpup, Dpdown, Tpup, Tpdown;

	public double[] a, b, c, ap, bp, cp; //latent features vectors and auxiliary vectors
	public double[] apup, apdown, bpup, bpdown, cpup, cpdown; 
	
	public double[] alpha, beta, gamma, rho, nu, delta; // Augmented coefficients
	public double[][] M, N, Z; // Lagrangian multipliers matrices
	public double[] p, q, h;  //Lagrangian multipliers vectors  
	
	public double lambda = 0;  //A constant adjusting the augmented coefficient
	public double eta = 0;     //Learning rate for the dual gradient ascent
	
	public double sumTime = 0; //Total time cost per training	

	public String trainFile = null;
	public String testFile = null;
	public String separator = null;
	
	protected InitTensor(String trainFile, String testFile, String separator)
	{
		this.trainFile = trainFile;
		this.testFile = testFile;
		this.separator = separator; 
	}
	
	public void initData(String inputFile, ArrayList<TensorTuple> data, int T) throws IOException
	{
		
		File input = new File(inputFile);
		BufferedReader in = new BufferedReader(new FileReader(input));
		String inTemp;
		while((inTemp = in.readLine()) != null)
		{
			StringTokenizer st = new StringTokenizer(inTemp, separator);
			
			String iTemp = null;
			if(st.hasMoreTokens())
				iTemp = st.nextToken();
			
			String jTemp = null;
			if(st.hasMoreTokens())
				jTemp = st.nextToken();
			
			String kTemp = null;
			if(st.hasMoreTokens())
				kTemp = st.nextToken();
			
			String tValueTemp = null;
			if(st.hasMoreTokens())
				tValueTemp = st.nextToken();
				
			int aid = Integer.valueOf(iTemp);
			int bid = Integer.valueOf(jTemp);
			int cid = Integer.valueOf(kTemp);
			double value = Double.valueOf(tValueTemp);

			this.maxAID = (this.maxAID > aid) ? this.maxAID : aid;
			this.maxBID = (this.maxBID > bid) ? this.maxBID : bid;
			this.maxCID = (this.maxCID > cid) ? this.maxCID : cid;
				
				
			this.minAID = (this.minAID < aid) ? this.minAID : aid;
			this.minBID = (this.minBID < bid) ? this.minBID : bid;
			this.minCID = (this.minCID < cid) ? this.minCID : cid;
				
			this.maxValue = (this.maxValue > value) ? this.maxValue : value;			
			this.minValue = (this.minValue < value) ? this.minValue : value;
				
			if(T == 1)
			{	
				this.testCount++;	 
			}
			else
			{
				this.trainCount++;
			}
				
 
			TensorTuple qtemp = new TensorTuple();
			qtemp.aID = aid;
			qtemp.bID = bid;
			qtemp.cID = cid;
			qtemp.value = value;
			
			data.add(qtemp);				
		}
	
		in.close();

	}


	
	public void logNormalization()
	{
		for(TensorTuple trainTuple: trainData)
		{
			trainTuple.value = Math.log10(trainTuple.value+1);
		}
		
		for(TensorTuple testTuple: testData)
		{
			testTuple.value = Math.log10(testTuple.value+1);
		}
	}

	
	public int scale = 1000;
	public double initscale = 0.001;
	
	public void initFactorMatrix() {
		S = new double[this.maxAID + 1][this.rank+1];
		D = new double[this.maxBID + 1][this.rank+1];
		T = new double[this.maxCID + 1][this.rank+1];
		
		Sp = new double[this.maxAID + 1][this.rank+1];
		Dp = new double[this.maxBID + 1][this.rank+1];
		Tp = new double[this.maxCID + 1][this.rank+1];
		
		a = new double[this.maxAID + 1];
		b = new double[this.maxBID + 1];
		c = new double[this.maxCID + 1];
		
		ap = new double[this.maxAID + 1];
		bp = new double[this.maxBID + 1];
		cp = new double[this.maxCID + 1];
		
		everyRoundRMSE = new double[this.trainRound+1];
		everyRoundMAE = new double[this.trainRound+1]; 
		
		everyRoundRMSE[0] = minRMSE;
		everyRoundMAE[0] = minMAE;
		
		Random random = new Random();
		for (int a_id = 1; a_id <= maxAID; a_id++) {
			for (int r = 1; r <= rank; r++) {
				Sp[a_id][r] = initscale * random.nextInt(scale) / scale; 
			}
			
			ap[a_id] = initscale * random.nextInt(scale) / scale; 
		}
		
		for (int b_id = 1; b_id <= maxBID; b_id++) {
			for (int r = 1; r <= rank; r++) {
				Dp[b_id][r] = initscale * random.nextInt(scale) / scale; 
			}
			
			bp[b_id] = initscale * random.nextInt(scale) / scale;
		}
		
		for (int c_id = 1; c_id <= maxCID; c_id++) {
			for (int r = 1; r <= rank; r++) {
				Tp[c_id][r] = initscale * random.nextInt(scale) / scale; 
			}
			
			cp[c_id] = initscale * random.nextInt(scale) / scale;
		}
	}
 
	
	public void initLagAssit() {

		M = new double[this.maxAID + 1][this.rank+1];
		N = new double[this.maxBID + 1][this.rank+1];
		Z = new double[this.maxCID + 1][this.rank+1];
		
		p = new double[this.maxAID + 1];
		q = new double[this.maxBID + 1];
		h = new double[this.maxCID + 1];
		
		alpha = new double[this.maxAID + 1];
		beta = new double[this.maxBID + 1];
		gamma = new double[this.maxCID + 1];
		rho = new double[this.maxAID + 1]; 
		nu = new double[this.maxBID + 1]; 
		delta = new double[this.maxBID + 1]; 
		
		Random random = new Random();
		
		for (int a_id = 1; a_id <= maxAID; a_id++) {
			
			alpha[a_id] = lambda * aSliceSet[a_id];
			rho[a_id] = lambda * aSliceSet[a_id];
			p[a_id] = initscale * random.nextInt(scale) / scale;;
			
			for (int r = 1; r <= rank; r++) {
				M[a_id][r] = initscale * random.nextInt(scale) / scale;;
			}
		}
		
		for (int b_id = 1; b_id <= maxBID; b_id++) {
			
			beta[b_id] = lambda * bSliceSet[b_id];
			nu[b_id] = lambda * bSliceSet[b_id];
			
			q[b_id] = initscale * random.nextInt(scale) / scale;;
			
			for (int r = 1; r <= rank; r++) {
				N[b_id][r] = initscale * random.nextInt(scale) / scale;;
			}
		}
		
		for (int c_id = 1; c_id <= maxCID; c_id++) {
			
			gamma[c_id] =  lambda * cSliceSet[c_id];
			delta[c_id] =  lambda * cSliceSet[c_id];
			
			h[c_id] = initscale * random.nextInt(scale) / scale;;
			
			for (int r = 1; r <= rank; r++) {
				Z[c_id][r] = initscale * random.nextInt(scale) / scale;;
			}
		}
		
	}

	
	public void initSliceSet() {
		aSliceSet = new double[maxAID + 1];
		bSliceSet = new double[maxBID + 1];
		cSliceSet = new double[maxCID + 1];
		for (TensorTuple tensor_tuple : trainData) {
			aSliceSet[tensor_tuple.aID] += 1;
			bSliceSet[tensor_tuple.bID] += 1;
			cSliceSet[tensor_tuple.cID] += 1;
		}
		
		for (int a_id = 1; a_id <= maxAID; a_id++) 
		{
			if(aSliceSet[a_id] == 0)
				aSliceSet[a_id] = 1;
		}
		
		for (int b_id = 1; b_id <= maxBID; b_id++) 
		{
			if(bSliceSet[b_id] == 0)
				bSliceSet[b_id] = 1;
		}
		
		for (int c_id = 1; c_id <= maxCID; c_id++) 
		{
			if(cSliceSet[c_id] == 0)
				cSliceSet[c_id] = 1;
		}

	}



	public void initAssistMatrix() {
		
		Spup = new double[maxAID + 1][this.rank+1];
		Spdown = new double[maxAID + 1][this.rank+1];
		Dpup = new double[maxBID + 1][this.rank+1];
		Dpdown = new double[maxBID + 1][this.rank+1];
		Tpup = new double[maxCID + 1][this.rank+1];
		Tpdown = new double[maxCID + 1][this.rank+1];
		
		apup = new double[maxAID + 1];
		apdown = new double[maxAID + 1];
		bpup = new double[maxBID + 1];
		bpdown = new double[maxBID + 1];
		cpup = new double[maxCID + 1];
		cpdown = new double[maxCID + 1];
		
		for (int max_a_id = 1; max_a_id <= maxAID; max_a_id++) 
		{
			for (int r = 1; r <= rank; r++) 
			{
				Spup[max_a_id][r] = 0;
				Spdown[max_a_id][r] = 0;
			}
			
			apup[max_a_id] = 0;
			apdown[max_a_id] = 0;
		}

		for (int max_b_id = 1; max_b_id <= maxBID; max_b_id++) 
		{
			for (int r = 1; r <= rank; r++) 
			{
				Dpup[max_b_id][r] = 0;
				Dpdown[max_b_id][r] = 0;
			}
			
			bpup[max_b_id] = 0;
			bpdown[max_b_id] = 0;
		}

		for (int max_c_id = 1; max_c_id <= maxCID; max_c_id++) 
		{
			for (int r = 1; r <= rank; r++) 
			{
				Tpup[max_c_id][r] = 0;
				Tpdown[max_c_id][r] = 0;
			}
			
			cpup[max_c_id] = 0;
			cpdown[max_c_id] = 0;
		}
		
	}

	

        public double getPredictionADMM(int a_Id, int b_Id, int c_Id) {
		double p_valueHat = 0;
		for (int r = 1; r <= this.rank; r++) {	
			p_valueHat += S[a_Id][r] * D[b_Id][r] * T[c_Id][r];
		}
		
		p_valueHat += (a[a_Id] + b[b_Id] + c[c_Id]);
		
		return p_valueHat;
	}
	
		
	public double getPrediction(int a_Id, int b_Id, int c_Id) {
		double p_valueHat = 0;
		for (int r = 1; r <= this.rank ; r++) {
			p_valueHat += Sp[a_Id][r] * Dp[b_Id][r] * Tp[c_Id][r];
		}	
		
		p_valueHat += (ap[a_Id] + bp[b_Id] + cp[c_Id]);
		
		return p_valueHat;
	}
}
