#ifndef FLOCK_CUH
#define FLOCK_CUH

#include "cuda_runtime.h"
#include <iostream>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


enum ExecutionType
{
	CPU,
	GPU
};

#define threadNum 32
#define VERTICES_PER_BOID_QUALITY 108
#define VERTICES_PER_BOID_PERFORMANCE 9



struct FlockProperties
{
	int numOfBoids = 10000;

	float alignmentRadius = 2.0f;
	float cohesionRadius = 2.0f;
	float separationRadius = 0.4f;

	float alignmentForce = 0.5f;
	float cohesionForce = 0.5f;
	float separationForce = 0.5f;

	float length = 0.2f;
	float width = 0.1f;

	float maxSpeed = 5.0f;
	float maxSteer = 2.0f;

	float3 color;
};


struct PoolProperties
{
	float Width = 20.0f;
	float Height = 20.0f;
	float Depth = 20.0f;
};


struct Grid
{
	int Height;
	int Width;
	int Depth;
	float CellSize;
	int* cellStartIdx;
	int* cellEndIdx;
};

void ReadPropertiesFromSetup(std::string SetupFileName, Grid& grid, FlockProperties& flocksProperties, PoolProperties& poolProperties);

void Init(Grid& grid, PoolProperties properties);

void Free(Grid& grid);
#endif
