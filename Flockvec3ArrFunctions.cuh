#ifndef FLOCKVEC3ARRFUNC_CUH
#define FLOCKVEC3ARRFUNC_CUH

#include "Flockvec3Arr.cuh"
#include "thrust/sort.h"
#include "thrust/device_ptr.h"

#include <stdio.h>
#include <utility>
#include<math.h>
#include <algorithm>
#include "helper_math.cuh"
#include "mat3.cuh"


__global__ void FindCellsStartEnd(int* sortedBoidGridPos, int* cellStartIdx, int* cellEndIdx, int numOfBoids);

__global__ void SetValues(int* array, int arraySize, int value);

__global__ void PrepareBoidIndexArray(int* boidIndex, int numOfBoids);

__device__ int3 IndexToGridPos(int Idx, Grid grid);

__device__ int GridPosToIdx(int3 gridPos, Grid grid);

__host__ float GenerateRandomFloat(float max);

__host__ __device__ float3 Limit(float3 steerVector, float value);

__device__ float3 CalculateSteerVector(float3 newVelocityVector, float3 oldVelocityVector, float maxSpeed, float maxSteer);

__host__ __device__ float3 CalculateGoodPosition(float3 position, PoolProperties poolProperties);

__host__ __device__ float3 AvoidWalls(float3 curPosition, float3 curVelocity, PoolProperties poolProperties, float deltaTime);

__global__ void CalculateGridCell(vec3Arr position, int* boidGridPosition, float gridCellSize, int gridHeight, int gridWidth, int numOfBoids);

__global__ void RearrangePosAndVel(int* boidIndex, Flockvec3Arr flock, int numOfBoids);

__global__ void UpdateBoids(Flockvec3Arr flock, Grid grid, FlockProperties flocksProperties, PoolProperties poolProperties, float deltaTime);

int Step(Flockvec3Arr& flock, Grid& grid, FlockProperties& flocksProperties, PoolProperties& poolProperties, float  deltaTime);

void StepCPU(Flockvec3Arr& flock, FlockProperties& flocksProperties, PoolProperties& poolProperties, float  deltaTime);

void Init(Flockvec3Arr& flock, FlockProperties& flockProperties, PoolProperties& poolProperties);

void InitCPU(Flockvec3Arr& flock, FlockProperties& flockProperties, PoolProperties& poolProperties);


void RandomSetup(vec3Arr& startPosition, vec3Arr& startVelocity, FlockProperties& flockProperties, PoolProperties& poolProperties);


// Source: https://gist.github.com/kevinmoran/b459801083e53edeb8a5a43c49f1341084
__device__ mat3 rotateAlign(float3 v1, float3 v2);


__device__ void GenerateBoidTriangle(float triangles[VERTICES_PER_BOID_QUALITY], float3 position, float3 velocity, float length, float width);

__device__ void GenerateBoidTrianglePerformance(float triangles[VERTICES_PER_BOID_PERFORMANCE], float3 position, float3 velocity, float length, float width);

__global__ void PrepareForDrawing(float* vertices, vec3Arr position, vec3Arr velocity, FlockProperties flockProperties);

__global__ void PrepareForDrawingPerformance(float* vertices, vec3Arr position, vec3Arr velocity, FlockProperties flockProperties);


int DrawBoids(Flockvec3Arr& flock, FlockProperties& flockProperties, float* vertices);

int DrawBoidsPerformance(Flockvec3Arr& flock, FlockProperties& flockProperties, float* vertices);

void DrawBoidsPerformanceCPU(Flockvec3Arr& flock, FlockProperties& flockProperties, float* vertices);


void CleanUp(Flockvec3Arr& flock, Grid& grid);




#endif