#include <eigen3/Eigen/Dense>
#include <random>
#include <cmath>
#include <iostream>
#include <functional>
#include <vector>

/**
 * Example structure of the Network (if nodes are all initalized with 3): 
 *
 * Input I (inputVec)	Input_Layer  Hidden_Layer  Output_Layer Output O (expectedVec) target
 *
 * inputData[0]			1 			 1  		   1 			expectedData[0]
 * 									 Whidden_output
 * 							  	
 * inputData[1]			2 			 2 			   2 			expectedData[1]
 * 						Winput_hidden
 * inputData[2]			3 			 3 			   3 			expectedData[2]
 *
 * 							Xhidden  Ohidden
 * 							  ->       ->
 * 							  			  Xoutput  Ooutput       
 * 							  			    ->       -> 		 
 * 							  			    Ehidden				Eoutput
 * 							  			      <-				   <-
 * 							  			 
 * 							            
 **/
class neuralNetwork {
	public:
		neuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningrate, 
				std::function<void(Eigen::MatrixXd&)> activationFunc = 
				[](Eigen::MatrixXd& data) {data = data.unaryExpr(
					[](double x){
					return 1/(1 + std::exp(-x));}
				);});
		void train(std::vector<double> inputData, std::vector<double> expectedData);
		std::vector<double> query(std::vector<double> inputData);

	private:
		// amount of nodes
		int inNodes, hiddNodes, outNodes;
		double learnR;

		// a matrix with all the weights from Input to Hidden
		// Winput_hidden
		// w11 w21
		// w12 w22
		// w12 = from Input-Node 1 to Hidden-Node 2
		Eigen::MatrixXd weightInToHidden;
		// a matrix with all the weights from Hidden to Output
		// Whidden_output
		Eigen::MatrixXd weightHiddenToOut;
		std::function<void(Eigen::MatrixXd&)> activation;

		Eigen::MatrixXd initWeight(int m, int n);
};
