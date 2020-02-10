#include "neuralNetwork.h"

neuralNetwork::neuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, 
		double learningrate, std::function<void(Eigen::MatrixXd&)> activationFunc):
	inNodes(inputNodes), hiddNodes(hiddenNodes), outNodes(outputNodes), learnR(learningrate)
{
	activation = activationFunc;
	weightInToHidden = initWeight(hiddNodes, inNodes);
	weightHiddenToOut = initWeight(outNodes, hiddNodes);
}

void neuralNetwork::train(std::vector<double> inputData, std::vector<double> expectedData) {
	Eigen::Map<Eigen::VectorXd> inputVector(inputData.data(), inputData.size());
	Eigen::Map<Eigen::VectorXd> expectedVector(expectedData.data(), expectedData.size());

	/* Calculate Input */
	// Calculate values that go into the Hidden_Layer
	// Xhidden = Winput_hidden * I
	Eigen::MatrixXd hidden = weightInToHidden * inputVector;
	// Calculate the Ohidden (output of the Hidden Layer Node) inside the Hidden_Layer node
	// Ohidden = sigmoid(Xhidden)
	activation(hidden);
	Eigen::MatrixXd hidden_outputs = hidden;

	// Calculate the values that go into the Output_Layer
	// Xoutput = Whidden_output * Ohidden
	hidden = weightHiddenToOut * hidden;
	// Calculate the output inside the Output Layer
	// Ooutput = sigmoid(Xoutput)
	activation(hidden);

	/* Calculate new Weights */
	// Eoutput = target - Ooutput
	Eigen::MatrixXd output_error = expectedVector - hidden;
	// split error by weights 
	// Ehidden = Whidden_output * Eoutput
	Eigen::MatrixXd hidden_errors = weightHiddenToOut.transpose() * output_error;

	// change weights depending on the error
	// deltaW = alpha * (Eoutput x target x (1 - target)) * Ooutput^T
	// tempLayer = (Eoutput x target x (1 - target)) ; (x = elementwise Multiplication)
	Eigen::VectorXd tempLayer = learnR * output_error.array() * 
			expectedVector.array() * (Eigen::VectorXd::Ones(outNodes) - expectedVector).array();
	Eigen::MatrixXd tempMatrix = tempLayer * hidden_outputs.transpose();
	weightHiddenToOut += tempMatrix;
	// deltaW = alpha * Ehidden x target x (1 - Ooutput) * I^T
	tempLayer = learnR * hidden_errors.array() * hidden_outputs.array() * 
			(Eigen::VectorXd::Ones(hiddNodes) - hidden_outputs).array();
	weightInToHidden = weightInToHidden + tempLayer * inputVector.transpose();


}

std::vector<double> neuralNetwork::query(std::vector<double> inputData) {
	Eigen::Map<Eigen::VectorXd> inputVector(inputData.data(), inputData.size());

	// Calculate values that go into the Hidden_Layer
	// Xhidden = Winput_hidden * I
	Eigen::MatrixXd hidden = weightInToHidden * inputVector;
	// Calculate the Ohidden (output of the Hidden Layer Node) inside the Hidden_Layer node
	// Ohidden = sigmoid(Xhidden)
	activation(hidden);

	// Calculate the values that go into the Output_Layer
	// Xoutput = Whidden_output * Ohidden
	hidden = weightHiddenToOut * hidden;
	// Calculate the output inside the Output Layer
	// Ooutput = sigmoid(Xoutput)
	activation(hidden);

	return std::vector<double>(hidden.data(), hidden.data() + outNodes);
}


Eigen::MatrixXd neuralNetwork::initWeight(int m, int n) {
	std::random_device rd;
	std::mt19937_64 gen{rd()};
	std::normal_distribution<> gaussien{0, 1/std::sqrt(m)};
	return Eigen::MatrixXd::NullaryExpr(m, n, [&](){return gaussien(gen);});
}
