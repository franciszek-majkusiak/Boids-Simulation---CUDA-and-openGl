#ifndef FLOCKVEC3ARR_CUH
#define FLOCKVEC3ARR_CUH
#include "device_launch_parameters.h"

#include "Flock.h"

struct vec3Arr
{
	float* x;
	float* y;
	float* z;
};

inline __host__ __device__ float3 GetValue(vec3Arr& arr, int i)
{
	return make_float3(arr.x[i], arr.y[i], arr.z[i]);
}

inline __host__ __device__ void SetValue(vec3Arr& arr, int i, float3 value)
{
	arr.x[i] = value.x;
	arr.y[i] = value.y;
	arr.z[i] = value.z;
}


void InitCPU(vec3Arr& arr, int n);
void FreeCPU(vec3Arr& arr);
void Init(vec3Arr& arr, int n);
void Free(vec3Arr& arr);
void Swap(vec3Arr& arr1, vec3Arr& arr2);
void CopyHostToDevice(vec3Arr& d_arr, vec3Arr& h_arr, int n);




struct Flockvec3Arr
{
	vec3Arr position;
	vec3Arr velocity;
	vec3Arr positionTmp;
	vec3Arr velocityTmp;
	int* boidGridPosition;
	int* boidIndex;
};

//void Init(Flockvec3Arr& flock, FlockProperties properties);
void Init(Flockvec3Arr& flock, int n);

void InitCPU(Flockvec3Arr& flock, int n);

void Free(Flockvec3Arr& flock);

void FreeCPU(Flockvec3Arr& flock);


#endif