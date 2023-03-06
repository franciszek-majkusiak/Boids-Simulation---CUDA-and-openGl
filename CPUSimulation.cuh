#include "Simulation.cuh"

class CPUSimulation : Simulation
{
public:
	CPUSimulation(std::string simulationString) : Simulation(simulationString)
	{
	}
	void Prepare() override
	{
		ReadPropertiesFromSetup(SimulationString, grid, flockProperties, poolProperties);
		camera = Camera3D(glm::vec3(poolProperties.Width / 2, poolProperties.Height / 2, 2 * poolProperties.Depth));
		cube = Cube(poolProperties.Width, poolProperties.Height, poolProperties.Depth);
		InitCPU(flock, flockProperties, poolProperties);

		glBindVertexArray(VAO_CUBE);
		glBindBuffer(GL_ARRAY_BUFFER, VBO_CUBE);
		glBufferData(GL_ARRAY_BUFFER, 72 * sizeof(float), cube.vertices, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		vertices = new float[VERTICES_PER_BOID_PERFORMANCE * flockProperties.numOfBoids];

		// build and compile shader program
		// ------------------------------------
		BoidShader = Shader("boid3DPerformance.vert", "boid3DPerformance.frag");
		cubeShader = Shader("cube3D.vert", "cube3D.frag");
	}

	void MainLoop() override
	{
		lastFrame = static_cast<float>(glfwGetTime());
		//cudaEventRecord(start);
		printf("start\n");
		// render loop
		// -----------
		while (!glfwWindowShouldClose(window))
		{
			// per-frame time logic
			// --------------------
			currentFrame = static_cast<float>(glfwGetTime());
			deltaTime = currentFrame - lastFrame;
			lastFrame = currentFrame;
			FPS = 1 / deltaTime;
			//cudaEventRecord(start);
			// input
			// -----
			processInput(window);

			// Mouse Mode
			// ------
			if (Settings)
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
				glfwSetCursorPosCallback(window, NULL);
				firstMouse = true;
			}
			else
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				glfwSetCursorPosCallback(window, mouse_callback);
			}


			// render
			// ------
			glClearColor(0.043f, 0.067f, 0.494f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 500.0f);
			glm::mat4 view = camera.GetViewMatrix();


			cubeShader.Activate();
			cubeShader.setMat4("projection", projection);
			cubeShader.setMat4("view", view);

			glBindVertexArray(VAO_CUBE);
			glDrawArrays(GL_LINES, 0, 24);
			glBindVertexArray(0);


			if (!isStopped)
			{
				cudaEventRecord(start);
				StepCPU(flock, flockProperties, poolProperties, deltaTime);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&UpdateTime, start, stop);
				UpdateTime /= 1000.0f;
				cudaEventRecord(start);
				DrawBoidsPerformanceCPU(flock, flockProperties, vertices);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&CreateVerticesTime, start, stop);
				CreateVerticesTime /= 1000.0f;
			}


			BoidShader.Activate();
			BoidShader.setMat4("projection", projection);
			BoidShader.setMat4("view", view);
			BoidShader.setVec3("color", flockProperties.color.x, flockProperties.color.y, flockProperties.color.z);


			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, VERTICES_PER_BOID_PERFORMANCE * flockProperties.numOfBoids * sizeof(float), vertices, GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDrawArrays(GL_TRIANGLES, 0, 3 * flockProperties.numOfBoids);
			glBindVertexArray(0);



			// ImGui
			// --------
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::Begin("Settings", 0, ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::Value("FPS", FPS);
			ImGui::Value("Frame Time", deltaTime);
			ImGui::Value("Update Time", UpdateTime);
			ImGui::Value("VerticesCreation Time", CreateVerticesTime);
			style->Colors[ImGuiCol_Text] = ImVec4(flockProperties.color.x, flockProperties.color.y, flockProperties.color.z, 1.0f);
			ImGui::Value("NumOfBoids", flockProperties.numOfBoids);
			ImGui::Text("Visability");
			ImGui::SliderFloat("Alignment Radius", &flockProperties.alignmentRadius, 0.0f, grid.CellSize, "%.3f");
			ImGui::SliderFloat("Cohesion Radius", &flockProperties.cohesionRadius, 0.0f, grid.CellSize, "%.3f");
			ImGui::SliderFloat("Separation Radius", &flockProperties.separationRadius, 0.0f, grid.CellSize, "%.3f");
			ImGui::Text("Steering");
			ImGui::SliderFloat("Speed", &flockProperties.maxSpeed, 0.0f, poolProperties.Width, "%.3f");
			ImGui::SliderFloat("Steer Force", &flockProperties.maxSteer, 0.0f, poolProperties.Width, "%.3f");
			ImGui::SliderFloat("Alignment Force", &flockProperties.alignmentForce, 0.0f, 1.0f, "%.3f");
			ImGui::SliderFloat("Cohesion Force", &flockProperties.cohesionForce, 0.0f, 1.0f, "%.3f");
			ImGui::SliderFloat("Separation Force", &flockProperties.separationForce, 0.0f, 1.0f, "%.3f");
			ImGui::Text("Boid Shape");
			ImGui::SliderFloat("Length", &flockProperties.length, 0.0f, 2.0f, "%.3f");
			ImGui::SliderFloat("Width", &flockProperties.width, 0.0f, 2.0f, "%.3f");
			ImGui::Text("Grid");
			ImGui::Value("Cell size: ", grid.CellSize);
			ImGui::Value("Width: ", grid.Width);
			ImGui::Value("Height: ", grid.Height);
			ImGui::Value("Depth: ", grid.Depth);
			if (ImGui::Button("Stop"))
			{
				isStopped = true;
			}
			if (ImGui::Button("Start"))
			{
				isStopped = false;
			}
			ImGui::End();

			//Render ImGui
			// ------------
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());



			// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
			// -------------------------------------------------------------------------------
			glfwSwapBuffers(window);
			glfwSwapInterval(0);
			glfwPollEvents();
		}
	}

	void CleanUp() override
	{
		Simulation::CleanUp();
		FreeCPU(flock);
		delete[] vertices;
	}
};