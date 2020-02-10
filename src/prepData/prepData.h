#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

class prepData {
	public:
		prepData(const std::string &trainingsDataDir, const std::string &testDataDir);
		std::vector<std::vector<double> > getTrainingsData();
		std::vector<std::vector<double> > getTestData();
		
	private:
		std::vector<std::string> getData(const std::string &filePath);
		std::string trainFilePath;
		std::string testFilePath;
		std::vector<double> split(const std::string &preparedData); 
};
