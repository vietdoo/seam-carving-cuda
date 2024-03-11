#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

// Define constant memory
#define FILTER_WIDTH 3
__constant__ float d_c_filter_x[FILTER_WIDTH * FILTER_WIDTH];
__constant__ float d_c_filter_y[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}


float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

void printError(uchar3 * deviceResult, uchar3 * hostResult, int width, int height)
{
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");
}


__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, 
		uint32_t * outPixels) {
	// TODO
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue 
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
 
  if (r < height && c < width) {
    int i = r * width + c;
    
    uint8_t red = inPixels[i].x;
    uint8_t green = inPixels[i].y;
    uint8_t blue = inPixels[i].z;
    outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
  } 
}

void convertRgb2Gray(uchar3 * inPixels, int width, int height, uint32_t * outPixels, 
		dim3 blockSize=dim3(1)) {

	uchar3 * d_in;
	uint32_t * d_out;
	CHECK(cudaMalloc(&d_in, width * height * sizeof(uchar3)));
  CHECK(cudaMalloc(&d_out, width * height * sizeof(uint32_t)));
	CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));
	
  dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
	convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_in, width, height, d_out);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    printf("ERROR: %s\n", cudaGetErrorString(err));
  }

	CHECK(cudaMemcpy(outPixels, d_out, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
}

__global__ void generateEnergyMatrixKernel(uint32_t * inPixels, int width, int height, uint32_t * outPixels) {

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    int s_width = blockDim.x + FILTER_WIDTH - 1;

    extern __shared__ uint32_t s_inPixels[];

    int i_row = r - FILTER_WIDTH / 2;
    int i_col = c - FILTER_WIDTH / 2;
    int prev_row = i_row;
    int prev_col = i_col;

    // Load data from inPixels to s_inPixels
    for (int s_row = threadIdx.y; s_row < s_width; s_row += blockDim.y) {
        prev_row = i_row;
        i_row = min(max(i_row, 0), height - 1);
        
        i_col = c - FILTER_WIDTH / 2;
        for (int s_col = threadIdx.x; s_col < s_width; s_col += blockDim.x) {
            prev_col = i_col;
            i_col = min(max(i_col, 0), width - 1);       
            s_inPixels[s_row * s_width + s_col] = inPixels[i_row * width + i_col];
            i_col = prev_col + blockDim.x;
        }
        i_row = prev_row + blockDim.y;
    } 
    __syncthreads();

    if (r < height && c < width)  {
      float out_x = 0;
      float out_y = 0;
      int filter_index = 0;
      for (int filter_row = 0; filter_row < FILTER_WIDTH; filter_row++) {
          for (int filter_col = 0; filter_col < FILTER_WIDTH; filter_col++) {
              int s_inPixels_index = (threadIdx.y + filter_row) * s_width + threadIdx.x + filter_col;
              out_x += s_inPixels[s_inPixels_index] * d_c_filter_x[filter_index];
              out_y += s_inPixels[s_inPixels_index] * d_c_filter_y[filter_index];
              filter_index = filter_index + 1;
          }
      }
      outPixels[r * width + c] = abs(out_x) + abs(out_y); 
    }
}

void generateEnergyMatrix(uint32_t * inPixels, int width, int height, 
      uint32_t * energyMatrix, dim3 blockSize=dim3(1)) {

    uint32_t * d_in, * d_energyMatrix;
    
    CHECK(cudaMalloc(&d_in, width * height * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_energyMatrix, width * height * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    size_t sharedMemSize = (blockSize.x + FILTER_WIDTH - 1) * (blockSize.y + FILTER_WIDTH - 1) * sizeof(uint32_t);
    generateEnergyMatrixKernel<<<gridSize, blockSize, sharedMemSize>>>(d_in, width, height, d_energyMatrix);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    
    CHECK(cudaMemcpy(energyMatrix, d_energyMatrix, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_energyMatrix));

}

__global__ void generateSeam(uint32_t * energy_matrix, int width, int curRow, 
uint32_t * min_energy_matrix, uint32_t * back_track_matrix) {

  int cur_col = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ uint32_t s_energy_matrix[];
  if (cur_col < width)
      s_energy_matrix[cur_col] = energy_matrix[curRow * width + cur_col];
  __syncthreads();

  if (cur_col < width) {

    int CUR_ID = curRow * width + cur_col;
    int UP_ROW_ID = (curRow - 1) * width + cur_col;
    int UP_CENTER_POS = UP_ROW_ID;
    int UP_LEFT_POS = UP_ROW_ID - 1;
    int UP_RIGHT_POS = UP_ROW_ID + 1;
    min_energy_matrix[CUR_ID] = s_energy_matrix[cur_col];

    if (cur_col == 0) {
      UP_LEFT_POS++;
    }
      
    if (cur_col == width - 1)
      UP_RIGHT_POS--;

    int min_pos = UP_CENTER_POS;
    int min_energy = min_energy_matrix[UP_CENTER_POS];

    if (min_energy_matrix[UP_RIGHT_POS] < min_energy) {
      min_pos = UP_RIGHT_POS;
      min_energy = min_energy_matrix[UP_RIGHT_POS];
    }

    if (min_energy_matrix[UP_LEFT_POS] < min_energy) {
      min_pos = UP_LEFT_POS;
      min_energy = min_energy_matrix[UP_LEFT_POS];
    }
    
    min_energy_matrix[CUR_ID] += min_energy;
    back_track_matrix[CUR_ID] = min_pos;
  }
}

void generateSeams(uint32_t * energy_matrix, int width, int height, 
      uint32_t * back_track_matrix,
      uint32_t * min_energy_matrix, dim3 blockSize = dim3(1)) {

  memcpy(min_energy_matrix, energy_matrix, width * sizeof(float));

  uint32_t * d_backtrack;
  CHECK(cudaMalloc(&d_backtrack, width * height * sizeof(uint32_t)));

  uint32_t * d_min_energy_matrix;
  uint32_t * d_energy_matrix;

  CHECK(cudaMalloc(&d_min_energy_matrix, width * height * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_energy_matrix, width * height * sizeof(uint32_t)));

  CHECK(cudaMemcpy(d_min_energy_matrix, min_energy_matrix, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_energy_matrix, energy_matrix, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));

  dim3 gridSize((width - 1) / blockSize.x + 1);

  for (int r = 1; r < height; r++){
    size_t sharedMemSize = width * sizeof(uint32_t);
    generateSeam<<<gridSize, blockSize, sharedMemSize>>>(d_energy_matrix, width, r, d_min_energy_matrix, d_backtrack);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
  }

  CHECK(cudaMemcpy(back_track_matrix, d_backtrack, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_backtrack));

  CHECK(cudaMemcpy(min_energy_matrix, d_min_energy_matrix, width * height* sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_min_energy_matrix));
  CHECK(cudaFree(d_energy_matrix));
}

void getSmallestSeam(uint32_t * min_energy_matrix,  int width, int height,
    uint32_t * seam, uint32_t * back_track_matrix) {
 
  int min_seam_sum = INT_MAX;
	int min_seam_id = 0;


  for (int c = 0; c < width; c++)  {
    int c_id = (height - 1) * width + c;
		if (min_energy_matrix[c_id] < min_seam_sum)  {
			min_seam_sum = min_energy_matrix[c_id];
			min_seam_id = c_id;
		}
	}
  
  memset(seam, 0, height * sizeof(uint32_t));
  seam[height - 1] = min_seam_id;
	for (int cur_row = height - 2; cur_row >= 0; cur_row--) {
    seam[cur_row] = back_track_matrix[min_seam_id];
    min_seam_id = back_track_matrix[min_seam_id];
	}
}

void removeSeam(uchar3 * input_matrix, uint32_t * seam, int width, int height) {
  uchar3 * crop_input_matrix = (uchar3 *)malloc((width - 1) * height * sizeof(uchar3));
  bool * seam_map = (bool *)malloc(width * height * sizeof(bool));
  memset(seam_map, 0, width * height * sizeof(bool));
  for (int i = 0; i < height; i++)
    seam_map[seam[i]] = 1;

  int crop_id = 0;
  for (int i = 0; i < height * width; i++)
  if (!seam_map[i]) {
    crop_input_matrix[crop_id++] = input_matrix[i];
  }
  input_matrix = (uchar3 *)realloc(input_matrix, (width - 1) * height * sizeof(uchar3));
  memcpy(input_matrix, crop_input_matrix, (width - 1) * height * sizeof(uchar3));
  free(crop_input_matrix);
  free(seam_map);
}

void seamCarving(uchar3 * inPixels, int width, int height, uchar3 * outPixels, int scale_width, 
        bool useDevice = false, dim3 blockSize= dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
  
  uchar3 * input_matrix = (uchar3 *)malloc(width * height * sizeof(uchar3));
  memcpy(input_matrix, inPixels, (width * height * sizeof(uchar3)));

  int cur_width = width;
  while (cur_width > scale_width) {
    
    uint32_t * gray_matrix = (uint32_t *)malloc(cur_width * height * sizeof(uint32_t));
    uint32_t * energy_matrix = (uint32_t *)malloc(cur_width * height * sizeof(uint32_t));
    uint32_t * seam = (uint32_t *)malloc(height * sizeof(uint32_t));
    uint32_t * back_track_matrix = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	  uint32_t * min_energy_matrix = (uint32_t *)malloc(width * height * sizeof(uint32_t));
    
    convertRgb2Gray(input_matrix, cur_width, height, gray_matrix, blockSize);
    generateEnergyMatrix(gray_matrix, cur_width, height, energy_matrix, blockSize);
    generateSeams(energy_matrix, cur_width, height, back_track_matrix, min_energy_matrix, blockSize);
    getSmallestSeam(min_energy_matrix, cur_width, height, seam, back_track_matrix);
    removeSeam(input_matrix, seam, cur_width, height);
    
    free(min_energy_matrix);
    free(back_track_matrix);
  	free(gray_matrix);
		free(energy_matrix);
    free(seam);

    cur_width = cur_width - 1;
  }

  memcpy(outPixels, input_matrix, scale_width * height * sizeof(uchar3));
  free(input_matrix);
	timer.Stop();
  float time = timer.Elapsed();
	printf("\nRun time: %f ms\n", time);
}

int main(int argc, char ** argv)
{
  float *filterX = new float[FILTER_WIDTH * FILTER_WIDTH]{-1, 0, 1,-2, 0, 2, -1, 0, 1};
  float *filterY = new float[FILTER_WIDTH * FILTER_WIDTH]{1, 2, 1, 0, 0, 0, -1, -2, -1};
  // Copy data to constant kernel
  CHECK(cudaMemcpyToSymbol(d_c_filter_x, filterX, FILTER_WIDTH * FILTER_WIDTH * sizeof(float)));
  CHECK(cudaMemcpyToSymbol(d_c_filter_y, filterY, FILTER_WIDTH * FILTER_WIDTH * sizeof(float)));

	if (argc != 3 && argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nInput image size (width x height): %i x %i\n", width, height);
  float scale_rate = 0.8;

  if (argc >= 4) 
  {
      scale_rate = atof(argv[3]);
  }
  int scale_width = width * scale_rate;
  printf("Output image size (width x height): %i x %i\n", scale_width, height);

  dim3 blockSize(32, 32);
  if (argc == 6)
  {
    blockSize.x = atoi(argv[4]);
    blockSize.y = atoi(argv[5]);
  }	

  printf("Block size: (%d, %d)\n", blockSize.x, blockSize.y);
	

	uchar3 * outPixels = (uchar3 *)malloc(scale_width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixels, scale_width, true, blockSize);
  char * outFileNameBase = strtok(argv[2], "."); 
	//writePnm(outPixels, scale_width, height, concatStr(outFileNameBase, "_device.pnm"));
  writePnm(outPixels, scale_width, height, concatStr("output/", "out_device_optimize_v1.pnm"));
	free(inPixels);
	free(outPixels);

}