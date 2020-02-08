#include <eigen3/Eigen/Dense>
#include <iostream>
#include "neuralNetwork.h"

int main() {
	neuralNetwork n(3, 3, 3, 0.8);
	std::vector<double> input{0.2, 0.3, 0.5};
	std::vector<double> result = n.query(input);
	for(auto v : result) {
		std::cout << v << std::endl;
	}
	return 0;
}
