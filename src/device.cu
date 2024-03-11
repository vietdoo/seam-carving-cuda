#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

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

__global__ void generateEnergyMatrixKernel(uint32_t * inPixels, int width, int height, 
		    float * filterX, float * filterY, int filter_width, uint32_t * outPixels) {

  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < height && c < width) {
    
    float out_x = 0, out_y = 0;
    int filter_row_index = 0;
    
    // Loop all 9 virtual pixels around target pixel and apply kernal
    for (int filter_row = r - filter_width / 2; filter_row < r + filter_width / 2 + 1; filter_row++) {
      for (int filter_col = c - filter_width / 2; filter_col < c + filter_width / 2 + 1; filter_col++) {
        int filter_row_fixed = filter_row;
        int filter_col_fixed = filter_col;

        if (filter_row < 0) {
          filter_row_fixed = 0;
        }
        if (filter_col < 0) {
          filter_col_fixed = 0;
        }
        if (filter_row > height - 1) {
          filter_row_fixed = height - 1;
        }
        if (filter_col > width - 1) {
          filter_col_fixed = width - 1;
        }

        int inPixels_index = filter_row_fixed * width + filter_col_fixed;
        
        out_x += inPixels[inPixels_index] * filterX[filter_row_index];
        out_y += inPixels[inPixels_index] * filterY[filter_row_index];
  
        filter_row_index = filter_row_index + 1;
        
      }
    }
  
    outPixels[r * width + c] = abs(out_x) + abs(out_y); 
  }
}


void generateEnergyMatrix(uint32_t * inPixels, int width, int height, 
      uint32_t * energyMatrix, dim3 blockSize=dim3(1)) {

    int filterWidth = 3;
    float *filterX = new float[filterWidth * filterWidth]{-1, 0, 1,-2, 0, 2, -1, 0, 1};
    float *filterY = new float[filterWidth * filterWidth]{1, 2, 1, 0, 0, 0, -1, -2, -1};
    float * d_filterX, * d_filterY;

    CHECK(cudaMalloc(&d_filterX, filterWidth * filterWidth * sizeof(float)));
    CHECK(cudaMalloc(&d_filterY, filterWidth * filterWidth * sizeof(float)));
    CHECK(cudaMemcpy(d_filterX, filterX, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filterY, filterY, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

    uint32_t * d_in, * d_energyMatrix;
    
    CHECK(cudaMalloc(&d_in, width * height * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_energyMatrix, width * height * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    
    generateEnergyMatrixKernel<<<gridSize, blockSize>>>(d_in, width, height, d_filterX, d_filterY, filterWidth, d_energyMatrix);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    
    CHECK(cudaMemcpy(energyMatrix, d_energyMatrix, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_filterX));
    CHECK(cudaFree(d_filterY));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_energyMatrix));

}

__global__ void generateSeam(uint32_t * energy_matrix, int width, int cur_row, 
uint32_t * min_energy_matrix, uint32_t * back_track_matrix) {

  int cur_col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (cur_col < width) {
    // Define all positions
    int CUR_ID = cur_row * width + cur_col;
    int UP_ROW_ID = (cur_row - 1) * width + cur_col;
    int UP_CENTER_POS = UP_ROW_ID;
    int UP_LEFT_POS = UP_ROW_ID - 1;
    int UP_RIGHT_POS = UP_ROW_ID + 1;

    // Init current sum is current energy point
    min_energy_matrix[CUR_ID] = energy_matrix[CUR_ID];

    // Check is outside image ?
    if (cur_col == 0) {
      UP_LEFT_POS++;
    }
      
    if (cur_col == width - 1)
      UP_RIGHT_POS--;

    // Init min position is up above (r - 1)
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
      uint32_t * back_track_matrix, uint32_t * min_energy_matrix, 
      dim3 blockSize = dim3(1)) {

  memcpy(min_energy_matrix, energy_matrix, width * sizeof(uint32_t));

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
    generateSeam<<<gridSize, blockSize>>>(d_energy_matrix, width, r, d_min_energy_matrix, d_backtrack);
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

  // Find smallet value at last row
  for (int c = 0; c < width; c++)  {
    int c_id = (height - 1) * width + c;
		if (min_energy_matrix[c_id] < min_seam_sum)  {
			min_seam_sum = min_energy_matrix[c_id];
			min_seam_id = c_id;
		}
	}
  
  memset(seam, 0, height * sizeof(uint32_t));

  // Back track to get seam (Auto reverse)
  seam[height - 1] = min_seam_id;
	for (int cur_row = height - 2; cur_row >= 0; cur_row--) {
    seam[cur_row] = back_track_matrix[min_seam_id];
    min_seam_id = back_track_matrix[min_seam_id];
	}
  
}

void removeSeam(uchar3 * input_matrix, uint32_t * seam, int width, int height) {
  // Create new image
  uchar3 * crop_input_matrix = (uchar3 *)malloc((width - 1) * height * sizeof(uchar3));

  // Create seam hash map to check in O(1)
  bool * seam_map = (bool *)malloc(width * height * sizeof(bool));
  memset(seam_map, 0, width * height * sizeof(bool));
  for (int i = 0; i < height; i++)
    seam_map[seam[i]] = 1;

  // Copy original img to new image that not contains seam path
  int crop_id = 0;
  for (int i = 0; i < height * width; i++)
  if (!seam_map[i]) {
    crop_input_matrix[crop_id++] = input_matrix[i];
  }
  

  // Reallocates for original image and copy back to original image
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

  int crop_width = width;
  // Loop until size equal scale_width
  while (crop_width > scale_width) {
    uint32_t * gray_matrix = (uint32_t *)malloc(crop_width * height * sizeof(uint32_t));
    uint32_t * energy_matrix = (uint32_t *)malloc(crop_width * height * sizeof(uint32_t));
    uint32_t * seam = (uint32_t *)malloc(height * sizeof(uint32_t));
    uint32_t * back_track_matrix = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	  uint32_t * min_energy_matrix = (uint32_t *)malloc(width * height * sizeof(uint32_t));
    
    convertRgb2Gray(input_matrix, crop_width, height, gray_matrix);
    generateEnergyMatrix(gray_matrix, crop_width, height, energy_matrix);
    generateSeams(energy_matrix, crop_width, height, back_track_matrix, min_energy_matrix);
    getSmallestSeam(min_energy_matrix, crop_width, height, seam, back_track_matrix);
    removeSeam(input_matrix, seam, crop_width, height);

		free(min_energy_matrix);
    free(back_track_matrix);
  	free(gray_matrix);
		free(energy_matrix);
    free(seam);

    crop_width = crop_width - 1;
  }


  memcpy(outPixels, input_matrix, scale_width * height * sizeof(uchar3));
  free(input_matrix);
	timer.Stop();
  float time = timer.Elapsed();
	printf("\nRun time: %f ms\n", time);
}

int main(int argc, char ** argv)
{
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
      blockSize.x = atoi(argv[3]);
      blockSize.y = atoi(argv[4]);
    }	
	

	uchar3 * outPixels = (uchar3 *)malloc(scale_width * height * sizeof(uchar3));
	seamCarving(inPixels, width, height, outPixels, scale_width, true, blockSize);
  //char * outFileNameBase = strtok(argv[2], "."); 
	//writePnm(outPixels, scale_width, height, concatStr(outFileNameBase, "_device.pnm"));
  writePnm(outPixels, scale_width, height, concatStr("output/", "out_device.pnm"));
	free(inPixels);
	free(outPixels);

}