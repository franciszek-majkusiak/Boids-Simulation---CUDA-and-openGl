#ifndef SHADER_H
#define SHADER_H

#include<glad/glad.h>
#include<string>
#include<fstream>
#include<sstream>
#include<iostream>
#include<cerrno>
#include<glm/glm.hpp>


std::string get_file_contents(const char* filename);


class Shader
{
public:
	unsigned int ID;
	Shader(const char* vertexFile, const char* fragmentFile);
	Shader();

	void Activate();
	void Delete();

	void setMat4(std::string uniformName, glm::mat4 unifromValue);
	void setFloat(std::string uniformName, float uniformValue);
	void setInt(std::string uniformName, int uniformValue);
	void setVec3(std::string uniformName, float x, float y, float z);
	void setVec3(std::string uniformName, glm::vec3 value);
};
#endif

