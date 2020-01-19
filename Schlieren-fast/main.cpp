#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#define STD_EXPORT
#define CSV_EXPORT
#define APP_CSV

#include "CL/cl2.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>


using namespace std;

void printDevice(int i, cl::Device& d)
{
	cout << "Device #" << i << endl;
	cout << "Name: " << d.getInfo<CL_DEVICE_NAME>() << endl;
	cout << "Type: " << d.getInfo<CL_DEVICE_TYPE>();
	cout << " (GPU = " << CL_DEVICE_TYPE_GPU << ", CPU = " << CL_DEVICE_TYPE_CPU << ")" << endl;
	cout << "Vendor: " << d.getInfo<CL_DEVICE_VENDOR>() << endl;
	cout << "Max Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
	cout << "Global Memory: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/(1024*1024) << " MByte" << endl;
	cout << "Max Clock Frequency: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << endl;
	cout << "Max Allocateable Memory: " << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/(1024*1024) << " MByte" << endl;
	cout << "Local Memory: " << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1024 << " KByte" << endl;
	cout << "Available: " << d.getInfo< CL_DEVICE_AVAILABLE>() << endl;


}

void print2D(uint8_t * out, int res)
{

	for (int i = 0; i < res; i++) {
		for (int j = 0; j < res; j++) {

			//cout << (int)Schlieren[i * Resolution + j];
			cout << ((out[i*res + j] == 1) ? '#' : ' ') << " ";

		}
		cout << endl;
	}

}

bool initOpenCL(cl::Device& dev, cl::Context& con, cl::Program& prog, cl::CommandQueue& q)
{

	cl::Platform platform = cl::Platform::getDefault();
	cout << "OpenCL version: " << platform.getInfo<CL_PLATFORM_VERSION>() << endl;

	if (platform() == 0) {
		cerr << "No OpenCL 2.0 platform found." << endl;
		return false;
	}

	vector<cl::Device> gpus;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);

	if (gpus.size() == 0) {
		cout << " No devices found. " << endl;
		return false;
	}

	for (int i = 0; i < gpus.size(); i++) {

		printDevice(i, gpus[i]);

	}

	unsigned int devid = 0;

	if (gpus.size() != 1) {
		cout << "Device choice: ";
		cin >> devid;
	}
	else {
		cout << "Choosing the only GPU" << endl;
		devid = 0;
	}

	if (devid >= 0 && devid < gpus.size()) {
		dev = gpus[devid];
	}
	else {
		cout << "Wrong device choice!" << endl;
		return false;
	}

	con = cl::Context({dev});

	cl::Program::Sources sources;

	ifstream sourcefile("kernel.cl");
	string sourcecode(istreambuf_iterator<char>(sourcefile), (istreambuf_iterator<char>()));

	sources.push_back({ sourcecode.c_str(), sourcecode.length() });

	prog = cl::Program (con, sources);
	if (prog.build({ dev }) != CL_SUCCESS) {
		cout << " Error building: " << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev) << endl;
		return false;
	}
	
	q = cl::CommandQueue (con, dev);

	return true;	   
}

cl::Device device;
cl::Context context;
cl::Program program;
cl::CommandQueue queue;

const int log2res = 10;
const int PseudoTiles = 16;
const long long int Resolution = (1 << log2res) * PseudoTiles;
const double Scale = 6.0;
const int Iteration = 1000;
const double Viewport_x = 0.0;
const double Viewport_y = 0.0;

void calculate(uint8_t * schlieren, long long int res = 32768, int iter = 1000, double scale = 6.0, double vx = 0.0, double vy = 0.0)
{
	
	cl::Buffer schlierenbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * res * res);

	cl::Kernel kernel = cl::Kernel(program, "schlieren");

	kernel.setArg(0, schlierenbuffer);
	kernel.setArg(1, scale);
	kernel.setArg(2, res);
	kernel.setArg(3, iter);
	kernel.setArg(4, vx);
	kernel.setArg(5, vy);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(res * res), cl::NullRange);
	queue.finish();

	queue.enqueueReadBuffer(schlierenbuffer, CL_TRUE, 0, sizeof(uint8_t) * Resolution * Resolution, schlieren);

}

void scaledown(uint8_t * schlieren_old, uint8_t * schlieren_new, int oldres)
{
	int newres = oldres / 2;
	cl::Buffer oldbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * oldres * oldres);
	cl::Buffer newbuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * newres * newres);

	cl::Kernel kernel = cl::Kernel(program, "scaledown");

	kernel.setArg(0, oldbuffer);
	kernel.setArg(1, newbuffer);
	kernel.setArg(2, oldres);

	queue.enqueueWriteBuffer(oldbuffer, CL_FALSE, 0, sizeof(uint8_t) * oldres * oldres, schlieren_old);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(newres * newres), cl::NullRange/*cl::NDRange(newres)*/);
	queue.finish();
	queue.enqueueReadBuffer(newbuffer, CL_TRUE, 0, sizeof(uint8_t) * newres * newres, schlieren_new);

}

int sumup(uint8_t * schlieren, int res)
{
	int count = 0;
	for (int i = 0; i < res; i++)
		for (int j = 0; j < res; j++)
			if (schlieren[j*res + i] == 1)
				count++;

	return count;
}

int main(int argc, char* argv[])
{
#ifdef CSV_EXPORT
#ifdef APP_CSV
	ofstream outfile("dim.csv", ios::out | ios::app);
#else
	ofstream outfile("dim.csv", ios::out);
#endif
#endif

	if (initOpenCL(device, context, program, queue)){
		cout << "Inited OpenCL successfully!" << endl;
	}
	else {
		cout << "Could not init OpenCL" << endl;
		return -1;
	}

	cout << "Calculating null sets..." << endl;

	uint8_t * SchlierenA = new uint8_t[Resolution * Resolution];
	uint8_t * SchlierenB = new uint8_t[Resolution / 2L * Resolution / 2L];

	uint8_t * Schlieren[] = { SchlierenA, SchlierenB };

	calculate(SchlierenA, Resolution, Iteration, Scale, Viewport_x, Viewport_y);

	cout << "Finished!" << endl;

	int res = Resolution;
	int N = 0;
#ifdef CSV_EXPORT
#ifndef APP_CSV
	outfile << "S;k;B;r;N;log r;log N" << endl;
#endif
#endif
	for (int i = 0; i < log2res ; i++) {
		cout << "Scaledown from " << res << " to " << res/2 << endl;
		N = sumup(Schlieren[i % 2], res);
		outfile << Scale << ";" << Iteration << ";" << res << ";" << res/Scale << ";" << N << ";" << log10(res/Scale) << ";" << log10(N) << endl;
		scaledown(Schlieren[i % 2], Schlieren[(i + 1) % 2], res);
		cout << "Scaledown & sumup finished..." << endl;
		res /= 2;

	}

#ifdef CSV_EXPORT
	outfile.close();
#endif

	delete[] SchlierenA, SchlierenB;

	cout << "Finished..." << endl;

	system("PAUSE");

	return 0;

}

