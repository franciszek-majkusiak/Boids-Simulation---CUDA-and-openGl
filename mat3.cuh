#ifndef MAT3_CUH
#define MAT3_CUH

#include "cuda_runtime.h"


struct mat3
{
	float r1c1;
	float r1c2;
	float r1c3;
	float r2c1;
	float r2c2;
	float r2c3;
	float r3c1;
	float r3c2;
	float r3c3;


	//float r1c1, float r1c2, float r1c3, float r2c1, float r2c2, float r2c3, float r3c1, float r3c2, float r3c3
	//float r1c1, float r2c1, float r3c1, float r1c2, float r2c2, float r3c2, float r1c3, float r2c3, float r3c3
};

inline __host__ __device__ mat3 make_mat3(float r1c1, float r1c2, float r1c3, float r2c1, float r2c2, float r2c3, float r3c1, float r3c2, float r3c3)
{
	mat3 outMat;
	outMat.r1c1 = r1c1;
	outMat.r1c2 = r1c2;
	outMat.r1c3 = r1c3;
	outMat.r2c1 = r2c1;
	outMat.r2c2 = r2c2;
	outMat.r2c3 = r2c3;
	outMat.r3c1 = r3c1;
	outMat.r3c2 = r3c2;
	outMat.r3c3 = r3c3;

	return outMat;
}

inline __host__ __device__ float3 operator*(mat3 mat, float3 vec)
{
	return make_float3(vec.x * mat.r1c1 + vec.y * mat.r2c1 + vec.z * mat.r3c1, vec.x * mat.r1c2 + vec.y * mat.r2c2 + vec.z * mat.r3c2, vec.x * mat.r1c3 + vec.y * mat.r2c3 + vec.z * mat.r3c3);
}

#endif