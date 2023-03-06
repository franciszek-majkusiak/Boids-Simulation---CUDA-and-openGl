#include "Flockvec3ArrFunctions.cuh"


__global__ void FindCellsStartEnd(int* sortedBoidGridPos, int* cellStartIdx, int* cellEndIdx, int numOfBoids)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numOfBoids) return;

	if (i == numOfBoids - 1)
	{
		cellEndIdx[sortedBoidGridPos[i]] = numOfBoids - 1;
	}
	else if (i == 0)
	{
		cellStartIdx[sortedBoidGridPos[i]] = 0;
	}
	else if (sortedBoidGridPos[i] != sortedBoidGridPos[i + 1])
	{
		cellEndIdx[sortedBoidGridPos[i]] = i;
		cellStartIdx[sortedBoidGridPos[i + 1]] = i + 1;
	}
}

__global__ void SetValues(int* array, int arraySize, int value)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= arraySize) return;
	array[i] = value;
}

__global__ void PrepareBoidIndexArray(int* boidIndex, int numOfBoids)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numOfBoids) return;
	boidIndex[i] = i;
}


__device__ int3 IndexToGridPos(int Idx, Grid grid)
{
	int offsetZ = grid.Width * grid.Height;
	int offsetY = grid.Width;
	int z = Idx / offsetZ;
	int y = (Idx - z * offsetZ) / offsetY;
	int x = Idx - z * offsetZ - y * offsetY;
	return make_int3(x, y, z);
}

__device__ int GridPosToIdx(int3 gridPos, Grid grid)
{
	int offsetZ = grid.Width * grid.Height;
	int offsetY = grid.Width;
	return gridPos.z * offsetZ + gridPos.y * offsetY + gridPos.x;
}

__host__ float GenerateRandomFloat(float max)
{
	float f = (float)rand() / (float)(RAND_MAX)*max;
	return f;
}

__device__ float3 Limit(float3 steerVector, float value)
{
	float R = sqrt(steerVector.x * steerVector.x + steerVector.y * steerVector.y + steerVector.z * steerVector.z);
	if (R < 0.0001f) R = 0.0001f;
	if (R > value)
	{
		return steerVector / R * value;
	}
	else return steerVector;
}

__device__ float3 CalculateSteerVector(float3 newVelocityVector, float3 oldVelocityVector, float maxSpeed, float maxSteer)
{
	newVelocityVector = Limit(newVelocityVector, maxSpeed) - oldVelocityVector;
	return Limit(newVelocityVector, maxSteer);
}

__host__ __device__ float3 CalculateGoodPosition(float3 position, PoolProperties poolProperties)
{
	if (position.x > poolProperties.Width) position.x = position.x - poolProperties.Width;
	if (position.x < 0) position.x = poolProperties.Width + position.x;
	if (position.y > poolProperties.Height) position.y = position.y - poolProperties.Height;
	if (position.y < 0) position.y = poolProperties.Height + position.y;
	if (position.z > poolProperties.Depth) position.z = position.z - poolProperties.Depth;
	if (position.z < 0) position.z = poolProperties.Depth + position.z;
	return position;
}

__device__ float3 AvoidWalls(float3 curPosition, float3 curVelocity, PoolProperties poolProperties, float deltaTime)
{
	float3 position = curPosition + deltaTime * curVelocity;
	if (position.x > poolProperties.Width || position.x < 0) curVelocity.x = -curVelocity.x;
	if (position.y > poolProperties.Height || position.y < 0) curVelocity.y = -curVelocity.y;
	if (position.z > poolProperties.Depth || position.z < 0) curVelocity.z = -curVelocity.z;
	return curVelocity;
}

__global__ void CalculateGridCell(vec3Arr position, int* boidGridPosition, float gridCellSize, int gridHeight, int gridWidth, int numOfBoids)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numOfBoids) return;

	float3 pos = GetValue(position, i);
	int gridPosX = (int)(pos.x / gridCellSize);
	int gridPosY = (int)(pos.y / gridCellSize);
	int gridPosZ = (int)(pos.z / gridCellSize);

	int gridPos = gridWidth * gridHeight * gridPosZ + gridPosY * gridWidth + gridPosX;
	boidGridPosition[i] = gridPos;
}


__global__ void RearrangePosAndVel(int* boidIndex, Flockvec3Arr flock, int numOfBoids)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numOfBoids) return;

	int idx = boidIndex[i];
	SetValue(flock.positionTmp, i, GetValue(flock.position, idx));
	SetValue(flock.velocityTmp, i, GetValue(flock.velocity, idx));
}


__global__ void UpdateBoids(Flockvec3Arr flock, Grid grid, FlockProperties flockProperties, PoolProperties poolProperties, float deltaTime)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= flockProperties.numOfBoids) return;

	float3 zero = make_float3(0.0f, 0.0f, 0.0f);
	float3 averagePosition = zero;
	float3 alignmentVector = zero;
	float3 separationVector = zero;
	float3 cohesionVector = zero;
	float3 currentBoidPosition = zero;
	float3 currentBoidVelocity = zero;
	float3 curCheckedBoidPosition = zero;
	float3 curCheckedBoidVelocity = zero;
	int numOfBoidsVisableAlignment = 0;
	int numOfBoidsVisableCohesion = 0;
	float RSQ = 0.0f;
	float3 dist = zero;
	currentBoidPosition = GetValue(flock.position, i);
	currentBoidVelocity = GetValue(flock.velocity, i);
	int gridIdx = flock.boidGridPosition[i];
	int3 gridPos = IndexToGridPos(gridIdx, grid);
	int testedCell = 0;
	int3 offSet = make_int3(0, 0, 0);
	int3 cellToCheck;
	int numOfTestedCell = 0;
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				offSet.x = x; offSet.y = y; offSet.z = z;
				cellToCheck = gridPos + offSet;
				if (cellToCheck.x < 0 || cellToCheck.x >= grid.Width || cellToCheck.y < 0 || cellToCheck.y >= grid.Height || cellToCheck.z < 0 || cellToCheck.z >= grid.Depth)
					continue;
				testedCell = GridPosToIdx(cellToCheck, grid);
				if (testedCell >= 0 && testedCell < grid.Width * grid.Height * grid.Depth)
				{
					numOfTestedCell++;
					if (grid.cellStartIdx[testedCell] != -1)
					{
						for (int j = grid.cellStartIdx[testedCell]; j <= grid.cellEndIdx[testedCell]; j++)
						{
							curCheckedBoidPosition = GetValue(flock.position, j);
							curCheckedBoidVelocity = GetValue(flock.velocity, j);
							dist = currentBoidPosition - curCheckedBoidPosition;
							RSQ = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
							if (RSQ < 0.0001f) RSQ = 0.0001f;
							if (j == i) continue;
							if (RSQ < flockProperties.alignmentRadius * flockProperties.alignmentRadius)
							{
								numOfBoidsVisableAlignment++;
								alignmentVector += curCheckedBoidVelocity;
							}
							if (RSQ < flockProperties.cohesionRadius * flockProperties.cohesionRadius)
							{
								numOfBoidsVisableCohesion++;
								averagePosition += curCheckedBoidPosition;
							}
							if (RSQ < flockProperties.separationRadius * flockProperties.separationRadius)
								separationVector += dist / RSQ;
						}
					}
				}
			}
		}
	}
	if (numOfBoidsVisableAlignment > 0)
		alignmentVector /= numOfBoidsVisableAlignment;
	if (numOfBoidsVisableCohesion > 0)
	{
		averagePosition /= numOfBoidsVisableCohesion;
		cohesionVector = (averagePosition - currentBoidPosition);
	}
	float3 steerVector = alignmentVector * flockProperties.alignmentForce + separationVector * flockProperties.separationForce + cohesionVector * flockProperties.cohesionForce;
	steerVector = Limit(steerVector, flockProperties.maxSteer);
	float3 newVelocity = currentBoidVelocity + steerVector;
	newVelocity = Limit(newVelocity, flockProperties.maxSpeed);
	newVelocity = AvoidWalls(currentBoidPosition, newVelocity, poolProperties, deltaTime);
	float3 newPosition = currentBoidPosition + deltaTime * newVelocity;
	newPosition = CalculateGoodPosition(newPosition, poolProperties);
	SetValue(flock.positionTmp, i, newPosition);
	SetValue(flock.velocityTmp, i, newVelocity);

}


int Step(Flockvec3Arr& flock, Grid& grid, FlockProperties& flockProperties, PoolProperties& poolProperties, float deltaTime)
{
	cudaError_t cudaStatus;
	int numOfBlocks = ceil((float)flockProperties.numOfBoids / (float)threadNum);
	CalculateGridCell << <numOfBlocks, threadNum >> > (flock.position, flock.boidGridPosition, grid.CellSize, grid.Height, grid.Width, flockProperties.numOfBoids);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CalculateGirCell launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CalculateGirCell!\n", cudaStatus);
		return -1;
	}

	PrepareBoidIndexArray << <numOfBlocks, threadNum >> > (flock.boidIndex, flockProperties.numOfBoids);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "PrepareBoidIndexArray launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching PrepareBoidIndexArray!\n", cudaStatus);
		return -1;
	}

	thrust::device_ptr<int> keys(flock.boidGridPosition);
	thrust::device_ptr<int> values(flock.boidIndex);
	thrust::sort_by_key(keys, keys + flockProperties.numOfBoids, values);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sort_by_key launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sort_by_key!\n", cudaStatus);
		return -1;
	}

	RearrangePosAndVel << < numOfBlocks, threadNum >> > (flock.boidIndex, flock, flockProperties.numOfBoids);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "RearrangePosAndVel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching RearrangePosAndVel!\n", cudaStatus);
		return -1;
	}
	Swap(flock.position, flock.positionTmp);
	Swap(flock.velocity, flock.velocityTmp);

	SetValues << <numOfBlocks, threadNum >> > (grid.cellStartIdx, flockProperties.numOfBoids, -1);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SetValues launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SetValues!\n", cudaStatus);
		return -1;
	}

	FindCellsStartEnd << < numOfBlocks, threadNum >> > (flock.boidGridPosition, grid.cellStartIdx, grid.cellEndIdx, flockProperties.numOfBoids);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FindCellsStartEnd  launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching FindCellsStartEnd !\n", cudaStatus);
		return -1;
	}

	UpdateBoids << <numOfBlocks, threadNum >> > (flock, grid, flockProperties, poolProperties, deltaTime);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "UpdateBoids launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching UpdateBoids!\n", cudaStatus);
		return -1;
	}
	Swap(flock.position, flock.positionTmp);
	Swap(flock.velocity, flock.velocityTmp);

	return 0;
}


void Init(Flockvec3Arr& flock, FlockProperties& flockProperties, PoolProperties& poolProperties)
{
	cudaSetDevice(0);
	cudaError_t cudaStatus;
	Init(flock, flockProperties.numOfBoids);

	vec3Arr startPosition;
	vec3Arr startVelocity;

	InitCPU(startPosition, flockProperties.numOfBoids);
	InitCPU(startVelocity, flockProperties.numOfBoids);
	RandomSetup(startPosition, startVelocity, flockProperties, poolProperties);

	CopyHostToDevice(flock.position, startPosition, flockProperties.numOfBoids);
	CopyHostToDevice(flock.velocity, startVelocity, flockProperties.numOfBoids);

	FreeCPU(startPosition);
	FreeCPU(startVelocity);
}


void RandomSetup(vec3Arr& startPosition, vec3Arr& startVelocity, FlockProperties& flockProperties, PoolProperties& poolProperties)
{
	srand(time(NULL));
	for (int i = 0; i < flockProperties.numOfBoids; i++)
	{
		SetValue(startPosition, i, make_float3(GenerateRandomFloat(poolProperties.Width), GenerateRandomFloat(poolProperties.Height), GenerateRandomFloat(poolProperties.Depth)));
		SetValue(startVelocity, i, normalize(make_float3(GenerateRandomFloat(1.0f), GenerateRandomFloat(1.0f), GenerateRandomFloat(1.0f))) * flockProperties.maxSpeed);
	}
}




__device__ void GenerateBoidTriangle(float triangles[VERTICES_PER_BOID_QUALITY], float3 position, float3 velocity, float length, float width)
{
	float3 direction = normalize(velocity);
	int counter = 0;
	mat3 rotationMat = rotateAlign(direction, make_float3(0.0f, 1.0f, 0.0f));
	float3 boidShape[18];

	//down
	boidShape[0] = make_float3(-width / 2, -length / 2, width / 2);
	boidShape[1] = make_float3(width / 2, -length / 2, width / 2);
	boidShape[2] = make_float3(-width / 2, -length / 2, -width / 2);

	//down
	boidShape[3] = make_float3(-width / 2, -length / 2, -width / 2);
	boidShape[4] = make_float3(width / 2, -length / 2, width / 2);
	boidShape[5] = make_float3(width / 2, -length / 2, -width / 2);

	//front
	boidShape[6] = make_float3(-width / 2, -length / 2, width / 2);
	boidShape[7] = make_float3(width / 2, -length / 2, width / 2);
	boidShape[8] = make_float3(0.0f, length / 2, 0.0f);


	//right
	boidShape[9] = make_float3(width / 2, -length / 2, width / 2);
	boidShape[10] = make_float3(width / 2, -length / 2, -width / 2);
	boidShape[11] = make_float3(0.0f, length / 2, 0.0f);

	//back
	boidShape[12] = make_float3(width / 2, -length / 2, -width / 2);
	boidShape[13] = make_float3(-width / 2, -length / 2, -width / 2);
	boidShape[14] = make_float3(0.0f, length / 2, 0.0f);

	//left
	boidShape[15] = make_float3(-width / 2, -length / 2, -width / 2);
	boidShape[16] = make_float3(-width / 2, -length / 2, width / 2);
	boidShape[17] = make_float3(0.0f, length / 2, 0.0f);

	float3 boidNormals[6];
	//down
	boidNormals[0] = make_float3(0.0f, -1.0f, 0.0f);
	boidNormals[1] = make_float3(0.0f, -1.0f, 0.0f);
	//front
	for (int i = 2; i < 6; i++)
	{
		boidNormals[i] = normalize(cross(boidShape[i * 3 + 1] - boidShape[i * 3], boidShape[i * 3 + 2] - boidShape[i * 3]));
	}
	for (int i = 0; i < 18; i++)
	{
		boidShape[i] = rotationMat * boidShape[i] + position;
		if (i < 6)
			boidNormals[i] = rotationMat * boidNormals[i];
		triangles[counter] = boidShape[i].x;
		triangles[counter + 1] = boidShape[i].y;
		triangles[counter + 2] = boidShape[i].z;
		int normalNum = i / 3;
		triangles[counter + 3] = boidNormals[normalNum].x;
		triangles[counter + 4] = boidNormals[normalNum].y;
		triangles[counter + 5] = boidNormals[normalNum].z;
		counter += 6;
	}
}



__global__ void PrepareForDrawing(float* vertices, vec3Arr position, vec3Arr velocity, FlockProperties flockProperties)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= flockProperties.numOfBoids) return;
	float boidTriangle[VERTICES_PER_BOID_QUALITY];
	GenerateBoidTriangle(boidTriangle, GetValue(position, i), GetValue(velocity, i), flockProperties.length, flockProperties.width);
	for (int j = 0; j < VERTICES_PER_BOID_QUALITY; j++)
	{
		vertices[i * VERTICES_PER_BOID_QUALITY + j] = boidTriangle[j];
	}
}

int DrawBoids(Flockvec3Arr& flock, FlockProperties& flockProperties, float* vertices)
{
	int numOfBlocks = ceil((float)flockProperties.numOfBoids / (float)threadNum);

	PrepareForDrawing << <numOfBlocks, threadNum >> > (vertices, flock.position, flock.velocity, flockProperties);
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "PrepareForDrawing launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching PrepareforDrawing!\n", cudaStatus);
		return -1;
	}
	return 0;
}






void CleanUp(Flockvec3Arr& flock, Grid& grid)
{
	Free(grid);
	Free(flock);
}

__device__ mat3 rotateAlign(float3 v1, float3 v2)
{
	float3 axis = cross(v1, v2);

	const float cosA = dot(v1, v2);
	const float k = 1.0f / (1.0f + cosA);
	mat3 rotationMat = make_mat3((axis.x * axis.x * k) + cosA,
		(axis.y * axis.x * k) - axis.z,
		(axis.z * axis.x * k) + axis.y,
		(axis.x * axis.y * k) + axis.z,
		(axis.y * axis.y * k) + cosA,
		(axis.z * axis.y * k) - axis.x,
		(axis.x * axis.z * k) - axis.y,
		(axis.y * axis.z * k) + axis.x,
		(axis.z * axis.z * k) + cosA);

	return rotationMat;
}



int DrawBoidsPerformance(Flockvec3Arr& flock, FlockProperties& flockProperties, float* vertices)
{
	int numOfBlocks = ceil((float)flockProperties.numOfBoids / (float)threadNum);

	PrepareForDrawingPerformance << <numOfBlocks, threadNum >> > (vertices, flock.position, flock.velocity, flockProperties);
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "PrepareForDrawing launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching PrepareforDrawing!\n", cudaStatus);
		return -1;
	}
	return 0;
}

__global__ void PrepareForDrawingPerformance(float* vertices, vec3Arr position, vec3Arr velocity, FlockProperties flockProperties)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= flockProperties.numOfBoids) return;
	float boidTriangle[VERTICES_PER_BOID_PERFORMANCE];
	GenerateBoidTrianglePerformance(boidTriangle, GetValue(position, i), GetValue(velocity, i), flockProperties.length, flockProperties.width);
	for (int j = 0; j < VERTICES_PER_BOID_PERFORMANCE; j++)
	{
		vertices[i * VERTICES_PER_BOID_PERFORMANCE + j] = boidTriangle[j];
	}
}

__host__ __device__ void GenerateBoidTrianglePerformance(float triangles[VERTICES_PER_BOID_PERFORMANCE], float3 position, float3 velocity, float length, float width)
{
	float3 direction = normalize(velocity);
	float3 front = position + length / 2 * direction;
	float3 back = position - length / 2 * direction;
	float3 directionRight = make_float3(direction.y, -direction.x, direction.y);
	float3 right = back + width / 2 * directionRight;
	float3 left = back - width / 2 * directionRight;
	triangles[0] = front.x;
	triangles[1] = front.y;
	triangles[2] = front.z;
	triangles[3] = right.x;
	triangles[4] = right.y;
	triangles[5] = right.z;
	triangles[6] = left.x;
	triangles[7] = left.y;
	triangles[8] = left.z;
}




///////CPU

void StepCPU(Flockvec3Arr& flock, FlockProperties& flockProperties, PoolProperties& poolProperties, float  deltaTime)
{
	for (int i = 0; i < flockProperties.numOfBoids; i++)
	{

		float3 zero = make_float3(0.0f, 0.0f, 0.0f);
		float3 averagePosition = zero;
		float3 alignmentVector = zero;
		float3 separationVector = zero;
		float3 cohesionVector = zero;
		float3 currentBoidPosition = zero;
		float3 currentBoidVelocity = zero;
		float3 curCheckedBoidPosition = zero;
		float3 curCheckedBoidVelocity = zero;
		int numOfBoidsVisableAlignment = 0;
		int numOfBoidsVisableCohesion = 0;
		float RSQ = 0.0f;
		float3 dist = zero;
		currentBoidPosition = GetValue(flock.position, i);
		currentBoidVelocity = GetValue(flock.velocity, i);
		for (int j = 0; j < flockProperties.numOfBoids; j++)
		{

			curCheckedBoidPosition = GetValue(flock.position, j);
			curCheckedBoidVelocity = GetValue(flock.velocity, j);
			dist = currentBoidPosition - curCheckedBoidPosition;
			RSQ = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
			if (RSQ < 0.0001f) RSQ = 0.0001f;
			if (j == i) continue;
			if (RSQ < flockProperties.alignmentRadius * flockProperties.alignmentRadius)
			{
				numOfBoidsVisableAlignment++;
				alignmentVector += curCheckedBoidVelocity;
			}
			if (RSQ < flockProperties.cohesionRadius * flockProperties.cohesionRadius)
			{
				numOfBoidsVisableCohesion++;
				averagePosition += curCheckedBoidPosition;
			}
			if (RSQ < flockProperties.separationRadius * flockProperties.separationRadius)
				separationVector += dist / RSQ;
		}
		if (numOfBoidsVisableAlignment > 0)
			alignmentVector /= numOfBoidsVisableAlignment;
		if (numOfBoidsVisableCohesion > 0)
		{
			averagePosition /= numOfBoidsVisableCohesion;
			cohesionVector = (averagePosition - currentBoidPosition);
		}
		float3 steerVector = alignmentVector * flockProperties.alignmentForce + separationVector * flockProperties.separationForce + cohesionVector * flockProperties.cohesionForce;
		steerVector = Limit(steerVector, flockProperties.maxSteer);
		float3 newVelocity = currentBoidVelocity + steerVector;
		newVelocity = Limit(newVelocity, flockProperties.maxSpeed);
		newVelocity = AvoidWalls(currentBoidPosition, newVelocity, poolProperties, deltaTime);
		float3 newPosition = currentBoidPosition + deltaTime * newVelocity;
		newPosition = CalculateGoodPosition(newPosition, poolProperties);
		SetValue(flock.positionTmp, i, newPosition);
		SetValue(flock.velocityTmp, i, newVelocity);
	}

	Swap(flock.position, flock.positionTmp);
	Swap(flock.velocity, flock.velocityTmp);
}


void InitCPU(Flockvec3Arr& flock, FlockProperties& flockProperties, PoolProperties& poolProperties)
{
	InitCPU(flock, flockProperties.numOfBoids);

	vec3Arr startPosition;
	vec3Arr startVelocity;

	RandomSetup(flock.position, flock.velocity, flockProperties, poolProperties);
}


void DrawBoidsPerformanceCPU(Flockvec3Arr& flock, FlockProperties& flockProperties, float* vertices)
{
	for (int i = 0; i < flockProperties.numOfBoids; i++)
	{
		float boidTriangle[VERTICES_PER_BOID_PERFORMANCE];
		GenerateBoidTrianglePerformance(boidTriangle, GetValue(flock.position, i), GetValue(flock.velocity, i), flockProperties.length, flockProperties.width);
		for (int j = 0; j < VERTICES_PER_BOID_PERFORMANCE; j++)
		{
			vertices[i * VERTICES_PER_BOID_PERFORMANCE + j] = boidTriangle[j];
		}
	}
	
}
