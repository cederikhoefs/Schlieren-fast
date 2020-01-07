#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#define STD_EXPORT
#define CSV_EXPORT
#undef APP_CSV

#include "CL/cl2.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>


using namespace std;

int main(int argc, char* argv[])
{
#ifdef CSV_EXPORT
#ifdef APP_CSV
	ofstream outfile("dim.csv", ios::out | ios::app);
#else
	ofstream outfile("dim.csv", ios::out);
#endif
#endif
	const int MaxResolution = 1000;
	const int ResolutionStep = 1000;
	const double Scale = 6.0;
	const int MaxIterations = 1000;
	const int IterationsStartStep = 1;
	const int IterationStep = 1000;
	const double Viewport_x = 0.0;
	const double Viewport_y = 0.0;


	cl::Platform platform = cl::Platform::getDefault();
	cout << platform.getInfo<CL_PLATFORM_VERSION>() << endl;

	if (platform() == 0) {
		std::cout << "No OpenCL 2.0 platform found." << endl;
		return -1;
	}

	vector<cl::Device> gpus;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);

	if (gpus.size() == 0) {
		std::cout << " No devices found. " << endl;
		return -1;
	}
	cl::Device default_device = gpus[0];
	cout << "Device: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;

	cl::Context context({ default_device });

	cl::Program::Sources sources;

	std::ifstream sourcefile("kernel.cl");
	std::string sourcecode(std::istreambuf_iterator<char>(sourcefile), (std::istreambuf_iterator<char>()));

	sources.push_back({ sourcecode.c_str(), sourcecode.length() });

	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << endl;
		return -1;
	}


	cl::CommandQueue queue(context, default_device);

#ifdef CSV_EXPORT
#ifdef APP_CSV
	//outfile << endl;
#else
	outfile << "Iterations;Scale;Resolution;N;dim" << endl;
#endif
#endif
	for (int i = IterationsStartStep; i <= MaxIterations / IterationStep; i++) {
		for (int j = 1; j <= MaxResolution / ResolutionStep; j++) {

			int Resolution = j * ResolutionStep; // MaxResolution
			int Iterations = i * IterationStep;
			
			bool* schlieren = new bool[Resolution * Resolution];

			cl::Buffer schlierenbuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * Resolution * Resolution);

			cl::Kernel kernel = cl::Kernel(program, "schlieren");

			kernel.setArg(0, schlierenbuffer);
			kernel.setArg(1, Scale);
			kernel.setArg(2, Resolution);
			kernel.setArg(3, Iterations);
			kernel.setArg(4, Viewport_x);
			kernel.setArg(5, Viewport_y);

			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(Resolution * Resolution), cl::NDRange(100));
			queue.finish();

			queue.enqueueReadBuffer(schlierenbuffer, CL_TRUE, 0, sizeof(bool) * Resolution * Resolution, schlieren);

			int belegt = 0;

			for (int i = 0; i < Resolution; i++) {
				for (int j = 0; j < Resolution; j++) {

					if (schlieren[i*Resolution + j]) {
						belegt++;
					}
				}
			}

#ifdef STD_EXPORT
			cout << "Iterations: " << Iterations << endl;
			cout << "Scale: " << Scale << endl;
			cout << "Resolution: " << Resolution << endl;
			cout << "N: " << belegt << endl;
			cout << "dim: " << -log(belegt) / log(Scale / Resolution) << endl;
#endif
#ifdef CSV_EXPORT
			outfile << Iterations << ";" << Scale << ";" << Resolution << ";" << belegt << ";" << -log(belegt) / log(Scale / Resolution) << endl;
#endif

			delete[] schlieren;

		}

	}
#ifdef CSV_EXPORT
	outfile.close();
#endif
	return 0;

}

