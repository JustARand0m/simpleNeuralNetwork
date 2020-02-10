#include <eigen3/Eigen/Dense>
#include <iostream>
#include "neuralNetwork/neuralNetwork.h"
#include "prepData/prepData.h"

int main() {
	prepData pData("../trainingsData/mnist_train_100.csv", "../trainingsData/mnist_test_10.csv");
	std::vector<std::vector<double> > trainingsData = pData.getTrainingsData();
	std::vector<std::vector<double> > testData = pData.getTestData();

	const int inputNodes = 784;
	const int hidden_nodes = 100;
	const int output_nodes = 10;
	const double learningrate = 0.3;
		
	neuralNetwork network(inputNodes, hidden_nodes, output_nodes, learningrate);

	for(int i = 0; i < trainingsData.size(); i++) {
		// get the expected Number, which is in pos 0 of the vector
		std::vector<double> result(10, 0.1);
		result[trainingsData[i][0]] = 0.99; 

		std::vector<double> test(trainingsData[i].begin() + 1, trainingsData[i].end());
		network.train(test, result);
	}

	

	return 0;
}
