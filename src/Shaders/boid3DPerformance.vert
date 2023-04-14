#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 view;
uniform mat4 projection;


void main()
{
	vec3 position = aPos;
	gl_Position = projection * view * vec4(position, 1.0);
}