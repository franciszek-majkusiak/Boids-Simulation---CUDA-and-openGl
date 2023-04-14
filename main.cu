#include "Quality.h"
#include "Performance.h"
#include "CPUSimulation.h"

int main()
{
	// Boid as 3d pyramid with phpng lighting model
	QualitySimulation simulation = QualitySimulation("SimpleTestFlock.setup");
	// Boid as 2d triangle  with no lighting
	//PerformanceSimulation simulation = PerformanceSimulation("BigFlock.setup");
	// same as PerformanceSymulation but boid calculations are done on CPU
	//CPUSimulation simulation = CPUSimulation("CPUFlock.setup");

	simulation.Prepare();
	simulation.MainLoop();
	simulation.CleanUp();
}