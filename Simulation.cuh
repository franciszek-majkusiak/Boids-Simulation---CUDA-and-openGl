#ifndef SIMULATION
#define SIMULATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <Windows.h>
#include <string>

#include "cuda_gl_interop.h"

#include "Shader.h"
#include "Camera3D.h"


#include "imgui\imgui.h"
#include "imgui\imgui_impl_glfw.h"
#include "imgui\imgui_impl_opengl3.h"


#include "Flockvec3ArrFunctions.cuh"
#include "Cube.cuh"

Camera3D camera;
static float lastX;
static float lastY;
static bool firstMouse;
static float deltaTime;

class Simulation
{
public:

	std::string SimulationString;


	Flockvec3Arr flock;
	FlockProperties flockProperties;
	Grid grid;
	PoolProperties poolProperties;

	Shader BoidShader;
	Shader cubeShader;

	Cube cube;

	GLFWwindow* window;

	bool Settings = false;
	bool wasReleased = true;
	bool wasPressed = false;

	float* vertices;
	cudaGraphicsResource* VBO_CUDA;

	unsigned int VAO, VBO, VAO_CUBE, VBO_CUBE;

	ImGuiIO io;
	ImGuiStyle* style;

	float lastFrame;
	float currentFrame;
	float UpdateTime = 0.0f, CreateVerticesTime = 0.0f;
	float FPS = 0.0f;

	int SCR_WIDTH, SCR_HEIGHT;

	cudaEvent_t start, stop;

	bool isStopped = false;

	glm::vec3 lightDir;

	Simulation(std::string simulationString)
	{
		SimulationString = simulationString;
		// Window setup
		// ------------
		SetupWindow(window);

		// glad: load all OpenGL function pointers
		// ---------------------------------------
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			exit(EXIT_FAILURE);
		}

		glEnable(GL_DEPTH_TEST);


		// ImGui Setup
		// -----------
		ImGui::CreateContext();
		io = ImGui::GetIO();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 130");
		ImGui::StyleColorsDark();


		glfwGetWindowSize(window, &SCR_WIDTH, &SCR_HEIGHT);
		lastX = SCR_WIDTH / 2.0f;
		lastY = SCR_HEIGHT / 2.0f;
		firstMouse = true;
		deltaTime = 0.0f;

		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO_CUBE);
		glGenBuffers(1, &VBO_CUBE);

		lightDir = glm::vec3(1.2f, 1.0f, 2.0f);

		style = &ImGui::GetStyle();

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	virtual void Prepare(){};
	virtual void MainLoop(){};
	virtual void CleanUp() 
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteVertexArrays(1, &VAO_CUBE);
		glDeleteBuffers(1, &VBO_CUBE);
		glfwTerminate();
		delete[] cube.vertices;
	};



	void SetupWindow(GLFWwindow*& window)
	{
		// glfw: initialize and configure
		// ------------------------------
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

		// glfw window creation
		// --------------------
		int count;
		GLFWmonitor** monitors = glfwGetMonitors(&count);
		GLFWmonitor* monitor = monitors[count - 1];
		const GLFWvidmode* mode = glfwGetVideoMode(monitor);

		window = glfwCreateWindow(mode->width, mode->height, "Boids", monitor, NULL);
		if (window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			exit(EXIT_FAILURE);
		}
		glfwMakeContextCurrent(window);
		glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
		glfwSetCursorPosCallback(window, (GLFWcursorposfun)mouse_callback);
		glfwSetScrollCallback(window, (GLFWscrollfun)scroll_callback);

		// tell GLFW to capture our mouse
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}


	void processInput(GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			camera.ProcessKeyboard(FORWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			camera.ProcessKeyboard(BACKWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			camera.ProcessKeyboard(LEFT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			camera.ProcessKeyboard(RIGHT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
			camera.ProcessKeyboard(UP, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
			camera.ProcessKeyboard(DOWN, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS && wasReleased)
		{
			if (Settings == true) Settings = false;
			else Settings = true;
			wasReleased = false;
		}
		if (glfwGetKey(window, GLFW_KEY_G) == GLFW_RELEASE)
			wasReleased = true;
	}


	// glfw: whenever the window size changed (by OS or user resize) this callback function executes
	// ---------------------------------------------------------------------------------------------
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		// make sure the viewport matches the new window dimensions; note that width and 
		// height will be significantly larger than specified on retina displays.
		glViewport(0, 0, width, height);
	}


	// glfw: whenever the mouse moves, this callback is called
	// -------------------------------------------------------
	static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
	{
		float xpos = static_cast<float>(xposIn);
		float ypos = static_cast<float>(yposIn);

		if (firstMouse)
		{
			lastX = xpos;
			lastY = ypos;
			firstMouse = false;
		}

		float xoffset = xpos - lastX;
		float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

		lastX = xpos;
		lastY = ypos;

		camera.ProcessMouseMovement(xoffset, yoffset);
	}

	// glfw: whenever the mouse scroll wheel scrolls, this callback is called
	// ----------------------------------------------------------------------
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
	{
		camera.ProcessMouseScroll(static_cast<float>(yoffset));
	}

	ImVec4 IntToColor(int color)
	{
		int r = (color & 0xff0000) >> 16;
		int g = (color & 0x00ff00) >> 8;
		int b = (color & 0x0000ff);

		return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
	}
};
#endif