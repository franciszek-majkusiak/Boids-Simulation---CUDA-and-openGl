//#include <glad/glad.h>
//#include <GLFW/glfw3.h>
//#include <iostream>
//#include <Windows.h>
//#include <string>
//
//#include "cuda_gl_interop.h"
//
//#include "Shader.h"
//#include "Camera3D.h"
//
//
//#include "imgui\imgui.h"
//#include "imgui\imgui_impl_glfw.h"
//#include "imgui\imgui_impl_opengl3.h"
//
//
//#include "Flockvec3ArrFunctions.cuh"
//#include "Cube.cuh"
//
//bool Settings = false;
//bool wasReleased = true;
//bool wasPressed = false;
//
//
//void getDesktopResolution(int* width, int* height)
//{
//	RECT desktop;
//
//	const HWND hDesktop = GetDesktopWindow();
//
//	GetWindowRect(hDesktop, &desktop);
//
//	*width = desktop.right;
//	*height = desktop.bottom;
//}
//
//
//ImVec4 IntToColor(int color);
//
//void SetupWindow(GLFWwindow*& window);
//
//void framebuffer_size_callback(GLFWwindow* window, int width, int height);
//void mouse_callback(GLFWwindow* window, double xpos, double ypos);
//void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
//void processInput(GLFWwindow* window);
//// settings
//const unsigned int SCR_WIDTH = 1920;
//const unsigned int SCR_HEIGHT = 1080;
//
//// camera3D
//Camera3D camera;
//float lastX = SCR_WIDTH / 2.0f;
//float lastY = SCR_HEIGHT / 2.0f;
//bool firstMouse = true;
//
//
//// timing
//float deltaTime = 0.0f;
//float lastFrame = 0.0f;
//
//// lighting
//glm::vec3 lightDir(1.2f, 1.0f, 2.0f);
//
//
//int main()
//{
//	// Window setup
//	// ------------
//	GLFWwindow* window;
//	SetupWindow(window);
//
//	// glad: load all OpenGL function pointers
//	// ---------------------------------------
//	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
//	{
//		std::cout << "Failed to initialize GLAD" << std::endl;
//		return -1;
//	}
//
//	glEnable(GL_DEPTH_TEST);
//
//
//	// ImGui Setup
//	// -----------
//	ImGui::CreateContext();
//	ImGuiIO& io = ImGui::GetIO();
//	ImGui_ImplGlfw_InitForOpenGL(window, true);
//	ImGui_ImplOpenGL3_Init("#version 130");
//	ImGui::StyleColorsDark();
//
//
//
//
//	// build and compile shader program
//	// ------------------------------------
//	Shader BoidShader("boid3D.vert", "boid3D.frag");
//	Shader cubeShader("cube3D.vert", "cube3D.frag");
//
//	PoolProperties poolProperties;
//	Grid grid;
//	FlockProperties flockProperties;
//	Flockvec3Arr flock;
//
//	ReadPropertiesFromSetup("C:\\Users\\fmajk\\Documents\\Sem5Studia\\PGwZO\\Boids Project\\BoidsFinal\\OneBigFlock.setup", grid, flockProperties, poolProperties);
//	camera = Camera3D(glm::vec3(poolProperties.Width / 2, poolProperties.Height / 2, 2 * poolProperties.Depth));
//
//	Cube cube(poolProperties.Width, poolProperties.Height, poolProperties.Depth);
//
//	Init(grid, poolProperties);
//	Init(flock, flockProperties, poolProperties);
//
//	float* vertices;
//	cudaGraphicsResource* VBO_CUDA;
//
//	unsigned int VAO, VBO, VAO_CUBE, VBO_CUBE;
//	glGenVertexArrays(1, &VAO);
//	glGenBuffers(1, &VBO);
//	glGenVertexArrays(1, &VAO_CUBE);
//	glGenBuffers(1, &VBO_CUBE);
//
//	glBindVertexArray(VAO_CUBE);
//	glBindBuffer(GL_ARRAY_BUFFER, VBO_CUBE);
//	glBufferData(GL_ARRAY_BUFFER, 72 * sizeof(float), cube.vertices, GL_STATIC_DRAW);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
//	glEnableVertexAttribArray(0);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//	glBindVertexArray(0);
//
//	//glBindVertexArray(VAO);
//	glBindBuffer(GL_ARRAY_BUFFER, VBO);
//	size_t size = flockProperties.numOfBoids * VERTICES_PER_BOID_COLOR * sizeof(float);
//	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//	//glBindVertexArray(0);
//	cudaGraphicsGLRegisterBuffer(&VBO_CUDA, VBO, cudaGraphicsMapFlagsWriteDiscard);
//	cudaGraphicsMapResources(1, &VBO_CUDA, 0);
//	cudaGraphicsResourceGetMappedPointer((void**)&vertices, &size, VBO_CUDA);
//
//	cudaEvent_t start, stop, startUpdate, stopUpdate;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventCreate(&startUpdate);
//	cudaEventCreate(&stopUpdate);
//
//	float currentFrame;
//	float UpdateTime = 0.0f, CopyTime = 0.0f;
//	float FPS = 0.0f;
//
//	ImGuiStyle* style = &ImGui::GetStyle();
//
//	lastFrame = static_cast<float>(glfwGetTime());
//	cudaEventRecord(start);
//	printf("start\n");
//	// render loop
//	// -----------
//	while (!glfwWindowShouldClose(window))
//	{
//		// per-frame time logic
//		// --------------------
//		currentFrame = static_cast<float>(glfwGetTime());
//		deltaTime = currentFrame - lastFrame;
//		lastFrame = currentFrame;
//		FPS = 1 / deltaTime;
//		cudaEventRecord(start);
//		// input
//		// -----
//		processInput(window);
//
//		// Mouse Mode
//		// ------
//		if (Settings)
//		{
//			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
//			glfwSetCursorPosCallback(window, NULL);
//			firstMouse = true;
//		}
//		else
//		{
//			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//			glfwSetCursorPosCallback(window, mouse_callback);
//		}
//
//
//		// render
//		// ------
//		glClearColor(0.043f, 0.067f, 0.494f, 1.0f);
//		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 500.0f);
//		glm::mat4 view = camera.GetViewMatrix();
//
//
//		cubeShader.Activate();
//		cubeShader.setMat4("projection", projection);
//		cubeShader.setMat4("view", view);
//
//		glBindVertexArray(VAO_CUBE);
//		glDrawArrays(GL_LINES, 0, 24);
//		glBindVertexArray(0);
//
//
//
//		cudaEventRecord(start);
//		Step(flock, grid, flockProperties, poolProperties, deltaTime);
//		cudaEventRecord(stop);
//		cudaEventSynchronize(stop);
//		cudaEventElapsedTime(&UpdateTime, start, stop);
//		UpdateTime /= 1000.0f;
//		cudaEventRecord(start);
//		DrawBoids(flock, flockProperties, vertices);
//		cudaEventRecord(stop);
//		cudaEventSynchronize(stop);
//		cudaEventElapsedTime(&CopyTime, start, stop);
//		CopyTime /= 1000.0f;
//		//CopyVerticesToHost(poolProperties, vertices);
//
//
//		BoidShader.Activate();
//		BoidShader.setMat4("projection", projection);
//		BoidShader.setMat4("view", view);
//		BoidShader.setVec3("viewPos", camera.Position);
//		BoidShader.setVec3("dirLight.direction", 0.0f, -1.0f, 0.0f);
//
//		glm::vec3 dirLightColor = glm::vec3(1.0, 1.0, 1.0);
//		glm::vec3 dirDiffuseColor = dirLightColor * glm::vec3(0.8f);
//		glm::vec3 dirAmbientColor = dirLightColor * glm::vec3(0.1f);
//		BoidShader.setVec3("dirLight.ambient", dirAmbientColor);
//		BoidShader.setVec3("dirLight.diffuse", dirDiffuseColor);
//		BoidShader.setVec3("dirLight.specular", 1.0f, 1.0f, 1.0f);
//
//
//		glBindVertexArray(VAO);
//		glBindBuffer(GL_ARRAY_BUFFER, VBO);
//
//		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
//		glEnableVertexAttribArray(0);
//		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
//		glEnableVertexAttribArray(1);
//		glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
//		glEnableVertexAttribArray(2);
//
//		glBindBuffer(GL_ARRAY_BUFFER, 0);
//		glDrawArrays(GL_TRIANGLES, 0, 18 * poolProperties.numOfAllBoids);
//		glBindVertexArray(0);
//
//
//
//		// ImGui
//		// --------
//		ImGui_ImplOpenGL3_NewFrame();
//		ImGui_ImplGlfw_NewFrame();
//		ImGui::NewFrame();
//
//		ImGui::Begin("Settings", 0, ImGuiWindowFlags_AlwaysAutoResize);
//		ImGui::Value("FPS", FPS);
//		ImGui::Value("Frame Time", deltaTime);
//		ImGui::Value("Update Time", UpdateTime);
//		ImGui::Value("Draw Time", CopyTime);
//		style->Colors[ImGuiCol_Text] = IntToColor(flockProperties.color);
//		ImGui::Value("NumOfBoids", flockProperties.numOfBoids);
//		ImGui::Text("Visability");
//		ImGui::SliderFloat("Alignment Radius", &flockProperties.alignmentRadius, 0.0f, grid.CellSize, "%.3f");
//		ImGui::SliderFloat("Cohesion Radius", &flockProperties.cohesionRadius, 0.0f, grid.CellSize, "%.3f");
//		ImGui::SliderFloat("Separation Radius", &flockProperties.separationRadius, 0.0f, grid.CellSize, "%.3f");
//		ImGui::Text("Steering");
//		ImGui::SliderFloat("Speed", &flockProperties.maxSpeed, 0.0f, poolProperties.Width, "%.3f");
//		ImGui::SliderFloat("Steer Force", &flockProperties.maxSteer, 0.0f, poolProperties.Width, "%.3f");
//		ImGui::SliderFloat("Alignment Force", &flockProperties.alignmentForce, 0.0f, 1.0f, "%.3f");
//		ImGui::SliderFloat("Cohesion Force", &flockProperties.cohesionForce, 0.0f, 1.0f, "%.3f");
//		ImGui::SliderFloat("Separation Force", &flockProperties.separationForce, 0.0f, 1.0f, "%.3f");
//		ImGui::Text("Boid Shape");
//		ImGui::SliderFloat("Length", &flockProperties.length, 0.0f, 2.0f, "%.3f");
//		ImGui::SliderFloat("Width", &flockProperties.width, 0.0f, 2.0f, "%.3f");
//		ImGui::Text("Grid");
//		ImGui::Value("Cell size: ", grid.CellSize);
//		ImGui::Value("Width: ", grid.Width);
//		ImGui::Value("Height: ", grid.Height);
//		ImGui::Value("Depth: ", grid.Depth);
//		ImGui::End();
//
//		//Render ImGui
//		// ------------
//		ImGui::Render();
//		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
//
//
//
//		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
//		// -------------------------------------------------------------------------------
//		glfwSwapBuffers(window);
//		glfwPollEvents();
//	}
//	glDeleteVertexArrays(1, &VAO);
//	glDeleteBuffers(1, &VBO);
//	glDeleteVertexArrays(1, &VAO_CUBE);
//	glDeleteBuffers(1, &VBO_CUBE);
//	// glfw: terminate, clearing all previously allocated GLFW resources.
//	// ------------------------------------------------------------------
//	glfwTerminate();
//	delete[] cube.vertices;
//	Free(flock);
//	Free(grid);
//	Free(poolProperties);
//	cudaGraphicsUnmapResources(1, &VBO_CUDA, 0);
//	return 0;
//}
//
//
//
//ImVec4 IntToColor(int color)
//{
//	int r = (color & 0xff0000) >> 16;
//	int g = (color & 0x00ff00) >> 8;
//	int b = (color & 0x0000ff);
//
//	return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
//}
//
//
//// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
//// ---------------------------------------------------------------------------------------------------------
//void processInput(GLFWwindow* window)
//{
//	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
//		glfwSetWindowShouldClose(window, true);
//
//	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
//		camera.ProcessKeyboard(FORWARD, deltaTime);
//	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
//		camera.ProcessKeyboard(BACKWARD, deltaTime);
//	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
//		camera.ProcessKeyboard(LEFT, deltaTime);
//	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
//		camera.ProcessKeyboard(RIGHT, deltaTime);
//	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
//		camera.ProcessKeyboard(UP, deltaTime);
//	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
//		camera.ProcessKeyboard(DOWN, deltaTime);
//	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS && wasReleased)
//	{
//		if (Settings == true) Settings = false;
//		else Settings = true;
//		wasReleased = false;
//	}
//	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_RELEASE)
//		wasReleased = true;
//}
//
//
//// glfw: whenever the window size changed (by OS or user resize) this callback function executes
//// ---------------------------------------------------------------------------------------------
//void framebuffer_size_callback(GLFWwindow* window, int width, int height)
//{
//	// make sure the viewport matches the new window dimensions; note that width and 
//	// height will be significantly larger than specified on retina displays.
//	glViewport(0, 0, width, height);
//}
//
//
//// glfw: whenever the mouse moves, this callback is called
//// -------------------------------------------------------
//void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
//{
//	float xpos = static_cast<float>(xposIn);
//	float ypos = static_cast<float>(yposIn);
//
//	if (firstMouse)
//	{
//		lastX = xpos;
//		lastY = ypos;
//		firstMouse = false;
//	}
//
//	float xoffset = xpos - lastX;
//	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
//
//	lastX = xpos;
//	lastY = ypos;
//
//	camera.ProcessMouseMovement(xoffset, yoffset);
//}
//
//// glfw: whenever the mouse scroll wheel scrolls, this callback is called
//// ----------------------------------------------------------------------
//void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
//{
//	camera.ProcessMouseScroll(static_cast<float>(yoffset));
//}
//
//void SetupWindow(GLFWwindow*& window)
//{
//	// glfw: initialize and configure
//	// ------------------------------
//	glfwInit();
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//
//#ifdef __APPLE__
//	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//#endif
//
//	// glfw window creation
//	// --------------------
//	int count;
//	GLFWmonitor** monitors = glfwGetMonitors(&count);
//	GLFWmonitor* monitor = monitors[count - 1];
//	const GLFWvidmode* mode = glfwGetVideoMode(monitor);
//
//	window = glfwCreateWindow(mode->width, mode->height, "Boids", monitor, NULL);
//	if (window == NULL)
//	{
//		std::cout << "Failed to create GLFW window" << std::endl;
//		glfwTerminate();
//		exit(EXIT_FAILURE);
//	}
//	glfwMakeContextCurrent(window);
//	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
//	glfwSetCursorPosCallback(window, mouse_callback);
//	glfwSetScrollCallback(window, scroll_callback);
//	int width;
//	int height;
//	getDesktopResolution(&width, &height);
//	glfwSetWindowPos(window, width / 2 - SCR_WIDTH / 2, height / 2 - SCR_HEIGHT / 2);
//
//	// tell GLFW to capture our mouse
//	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//}