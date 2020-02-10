#include "prepData.h"

prepData::prepData(const std::string &trainingsDataDir, const std::string &testDataDir) {
	trainFilePath = trainingsDataDir;
	testFilePath = testDataDir;
}


std::vector<std::vector<double> > prepData::getTrainingsData() {
	std::vector<std::string> preparedData = getData(trainFilePath);
	std::vector<std::vector<double> > finalImages;
	for(auto data : preparedData) {
		std::vector<double> image = split(data);
		finalImages.push_back(image);
	}
	return finalImages;
}


std::vector<std::vector<double> > prepData::getTestData() {
	std::vector<std::string> preparedData = getData(testFilePath);
	std::vector<std::vector<double> > finalImages;
	for(auto data : preparedData) {
		std::vector<double> image = split(data);
		finalImages.push_back(image);
	}
	return finalImages;
}



std::vector<std::string> prepData::getData(const std::string &filePath) {
	std::vector<std::string> preparedData;
	std::ifstream data(filePath);
	std::string line;
	while(std::getline(data, line)) {
		preparedData.push_back(line);	
	}
	data.close();
	return preparedData;

}


std::vector<double> prepData::split(const std::string &preparedData) {
	std::istringstream iss(preparedData);
	std::vector<double> pixels;
	std::string item;
	int i = 0; 
	while(std::getline(iss, item, ',')) {
		if(0 == i) {
			pixels.push_back(std::stod(item));
		} else {
			pixels.push_back((std::stod(item) / 255.0 * 0.99) + 0.01);
			i++;
		}
	}
	return pixels;
}
