#ifndef CUBE_H
#define CUBE_H
#include<stdio.h>
#include<iostream>

struct Cube
{
	float* vertices;
	Cube() {}
	Cube(float Width, float Height, float Depth)
	{
		vertices = new float[72];
		float tmpvertices[] = 
		{
			0.0f, 0.0f, 0.0f,
			Width, 0.0f, 0.0f,

			Width, 0.0f, 0.0f,
			Width, Height, 0.0f,

			Width, Height, 0.0f,
			0.0f, Height, 0.0f,

			0.0f, Height, 0.0f,
			0.0f, 0.0f, 0.0f,



			0.0f, 0.0f, Depth,
			Width, 0.0f, Depth,

			Width, 0.0f, Depth,
			Width, Height, Depth,

			Width, Height, Depth,
			0.0f, Height, Depth,

			0.0f, Height, Depth,
			0.0f, 0.0f, Depth,


			0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, Depth,

			Width, 0.0f, 0.0f,
			Width, 0.0f, Depth,

			Width, Height, 0.0f,
			Width, Height, Depth,

			0.0f, Height, 0.0f,
			0.0f, Height, Depth,
		};
		std::memcpy(vertices, tmpvertices, 72 * sizeof(float));
	}
};

#endif