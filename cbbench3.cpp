#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <cmath>

// -------------------- CPU Benchmark --------------------
void benchCPU(std::vector<float>& data, int target_seconds) {
    size_t N = data.size();
    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 0;

    while (true) {
        std::vector<std::thread> threads;
        size_t chunk = N / numThreads;
        for (unsigned t = 0; t < numThreads; ++t) {
            size_t begin = t * chunk;
            size_t end = (t == numThreads - 1) ? N : begin + chunk;
            threads.emplace_back([&data, begin, end](){
                for (size_t i = begin; i < end; ++i)
                    data[i] += 1.0f;
            });
        }

        for (auto& th : threads) th.join();
        iterations++;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - start).count() >= target_seconds)
            break;
    }

    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    double sum = std::accumulate(data.begin(), data.end(), 0.0);

    std::cout << "CPU benchmark complete (" << iterations << " iterations, "
              << numThreads << " threads): " << elapsed << " s\n";
    std::cout << "CPU checksum: " << sum << "\n";
}

// -------------------- GPU Benchmark (ALU + memory stress) --------------------
void benchGPU(size_t N, int target_seconds) {
    const char* kernelSource = R"CLC(
    __kernel void benchKernel(__global float* data, __global float* aux) {
        int idx = get_global_id(0);
        float x = data[idx] + idx * 0.001f;
        float y = aux[idx] + idx * 0.002f;

        for(int i = 0; i < 8; ++i) {
            x = sin(x) + cos(y*0.5f) + x*x*0.25f;      // ALU-heavy
            y = log(y + 1.0f) + sqrt(x);               // ALU + memory
            x += aux[(idx+i) % get_global_size(0)] * 0.0001f; // memory access
        }

        data[idx] = x;
        aux[idx] = y;
    }
    )CLC";

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

    cl_command_queue_properties props[] = {0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, nullptr);

    // Initialize buffers
    std::vector<float> data(N), aux(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = float(i) / N;
        aux[i] = float(i) / N * 2.0f;
    }

    cl_int err;
    cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * N, data.data(), &err);
    cl_mem d_aux  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * N, aux.data(), &err);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "benchKernel", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_aux);

    size_t globalSize = N;

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 0;

    while (true) {
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue);
        iterations++;

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - start).count() >= target_seconds)
            break;
    }

    // Read back results
    clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, sizeof(float) * N, data.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_aux, CL_TRUE, 0, sizeof(float) * N, aux.data(), 0, nullptr, nullptr);

    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) sum += data[i] + aux[i];

    std::cout << "GPU benchmark complete (" << iterations << " iterations): "
              << elapsed << " s\n";
    std::cout << "GPU checksum (dynamic, sanity check): " << sum << "\n";

    clReleaseMemObject(d_data);
    clReleaseMemObject(d_aux);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// -------------------- GPU Clock Readout --------------------
void printGpuClocksSmi() {
    std::cout << "GPU clocks via nvidia-smi:\n";

    FILE* pipe = _popen("nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits", "r");
    if (!pipe) {
        std::cout << "Failed to run nvidia-smi\n";
        return;
    }

    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer);
        std::istringstream iss(line);
        std::string coreStr, memStr;
        if (std::getline(iss, coreStr, ',') && std::getline(iss, memStr)) {
            try {
                int coreMHz = std::stoi(coreStr);
                int memMHz  = std::stoi(memStr);
                std::cout << "  Core:   " << coreMHz << " MHz\n";
                std::cout << "  Memory: " << memMHz  << " MHz\n";
            } catch (...) {
                std::cout << "Failed to parse GPU clocks\n";
            }
        }
    }

    _pclose(pipe);
}

// -------------------- Main --------------------
int main() {
    const size_t N = 16 * 1024 * 1024; // 16M elements
    const int bench_time = 15;         // seconds per benchmark

    std::vector<float> cpuData(N, 0.0f);
    benchCPU(cpuData, bench_time);

    std::cout << "\nGPU clocks BEFORE benchmark:\n";
    printGpuClocksSmi();

    benchGPU(N, bench_time);

    std::cout << "\nGPU clocks AFTER benchmark:\n";
    printGpuClocksSmi();

    return 0;
}
