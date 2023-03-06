#include "Quality.cuh"
#include "Performance.cuh"
#include "CPUSimulation.cuh"

int main()
{
	QualitySimulation simulation = QualitySimulation("SimpleTestFlock.setup");
	//PerformanceSimulation simulation = PerformanceSimulation("BigFlock.setup");
	//CPUSimulation simulation = CPUSimulation("CPUFlock.setup");

	simulation.Prepare();
	simulation.MainLoop();
	simulation.CleanUp();
}