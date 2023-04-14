#include "Flock.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>



void Init(Grid& grid, PoolProperties properties)
{
	printf("Init Grid\n");
	grid.Height = (int)ceil(properties.Height / grid.CellSize);
	grid.Width = (int)ceil(properties.Width / grid.CellSize);
	grid.Depth = (int)ceil(properties.Depth / grid.CellSize);

	int* cellStartIdx = 0;
	int* cellEndIdx = 0;

	checkCudaErrors(cudaMalloc((void**)&cellStartIdx, grid.Height * grid.Width * grid.Depth * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cellEndIdx, grid.Height * grid.Width * grid.Depth * sizeof(int)));

	grid.cellStartIdx = cellStartIdx;
	grid.cellEndIdx = cellEndIdx;
}

void Free(Grid& grid)
{
	printf("Free grid\n");
	cudaFree(grid.cellStartIdx);
	cudaFree(grid.cellEndIdx);
}


void split(const std::string& str, std::vector<std::string>& v)
{
	std::stringstream ss(str);
	ss >> std::noskipws;
	std::string field;
	char ws_delim;
	while (1) {
		if (ss >> field)
			v.push_back(field);
		else if (ss.eof())
			break;
		else
			v.push_back(std::string());
		ss.clear();
		ss >> ws_delim;
	}
}



void ReadPropertiesFromSetup(std::string SetupFileName, Grid& grid, FlockProperties& flockProperties, PoolProperties& poolProperties)
{
	std::string relativePath = "src/SetUpFiles/" + SetupFileName;
	std::ifstream infile(relativePath);
	if (!infile.is_open())
	{
		fprintf(stderr, "Cannot open file %s\n", relativePath.c_str());
		exit(EXIT_FAILURE);
	}
	std::string line;
	std::vector<std::string> lineSplitted;
	std::getline(infile, line);
	//Pool Width
	std::getline(infile, line);
	split(line, lineSplitted);
	poolProperties.Width = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	//Pool Height
	std::getline(infile, line);
	split(line, lineSplitted);
	poolProperties.Height = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	//Pool Depth
	std::getline(infile, line);
	split(line, lineSplitted);
	poolProperties.Depth = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	//Grid Cell Size
	std::getline(infile, line);
	split(line, lineSplitted);
	grid.CellSize = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.numOfBoids = std::stoi(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.alignmentRadius = grid.CellSize < std::stof(lineSplitted[1]) ? grid.CellSize : std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.cohesionRadius = grid.CellSize < std::stof(lineSplitted[1]) ? grid.CellSize : std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.separationRadius = grid.CellSize < std::stof(lineSplitted[1]) ? grid.CellSize : std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.alignmentForce = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.cohesionForce = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.separationForce = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.length = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.width = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.maxSpeed = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.maxSteer = std::stof(lineSplitted[1]);
	lineSplitted.clear();
	std::getline(infile, line);
	split(line, lineSplitted);
	flockProperties.color = make_float3(std::stof(lineSplitted[1]), std::stof(lineSplitted[2]), std::stof(lineSplitted[3]));
	lineSplitted.clear();
	infile.close();
}