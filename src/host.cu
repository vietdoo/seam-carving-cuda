#include <stdio.h>
#include <stdint.h>

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

void readPnm(char * fileName, 
		int &width, int &height, uchar3 * &pixels)
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

void writePnm(uchar3 * pixels, int width, int height, 
		char * fileName)
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

void convertRgb2Gray(uchar3 * inPixels, int width, int height, uint32_t * outPixels){
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      int i = r * width + c;
      float red = inPixels[i].x;
      float green = inPixels[i].y;
      float blue = inPixels[i].z;
      outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
  }
}

void generateEnergyMatrix(uint32_t * inPixels, int width, int height, uint32_t * energy_matrix)
{
  int filter_width = 3;
	float *filterX = new float[filter_width * filter_width]{-1, 0, 1,-2, 0, 2, -1, 0, 1};
  float *filterY = new float[filter_width * filter_width]{1, 2, 1, 0, 0, 0, -1, -2, -1};
	
	for (int c = 0; c < width; c++) {
    for (int r = 0; r < height; r++) {

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
      energy_matrix[r * width + c] = abs(out_x) + abs(out_y); 
    }
  }
}

void generateSeams(uint32_t * energy_matrix, int width, int height, 
      uint32_t * back_track_matrix,
      uint32_t * min_energy_matrix)  {

  memcpy(min_energy_matrix, energy_matrix, width * sizeof(uint32_t));

  for (int r = 1; r < height; r++)  {
    for (int c = 0; c < width; c++)  {
      // Define all positions
      int CUR_ID = r * width + c;
      int UP_ROW_ID = (r - 1) * width + c;
      int UP_CENTER_POS = UP_ROW_ID;
      int UP_LEFT_POS = UP_ROW_ID - 1;
      int UP_RIGHT_POS = UP_ROW_ID + 1;

      // Init current sum is current energy point
      min_energy_matrix[CUR_ID] = energy_matrix[CUR_ID];

      // Check is outside image ?
      if (c == 0)
        UP_LEFT_POS++;
        
      if (c == width - 1)
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


void seamCarving(uchar3 * inPixels, int width, int height, uchar3 * outPixels, int scale_width) {
	
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

void seamCarvingCalTime(uchar3 * inPixels, int width, int height, uchar3 * outPixels, int scale_width) {
	
  GpuTimer timer_sum;
	timer_sum.Start();
	
  uchar3 * input_matrix = (uchar3 *)malloc(width * height * sizeof(uchar3));
  memcpy(input_matrix, inPixels, (width * height * sizeof(uchar3)));
  
  float t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	
  int crop_width = width;
  // Loop until size equal scale_width
  while (crop_width > scale_width) {
    uint32_t * gray_matrix = (uint32_t *)malloc(crop_width * height * sizeof(uint32_t));
    uint32_t * energy_matrix = (uint32_t *)malloc(crop_width * height * sizeof(uint32_t));
    uint32_t * seam = (uint32_t *)malloc(height * sizeof(uint32_t));
    uint32_t * back_track_matrix = (uint32_t *)malloc(width * height * sizeof(uint32_t));
	  uint32_t * min_energy_matrix = (uint32_t *)malloc(width * height * sizeof(uint32_t));
    
    GpuTimer timer_sub;
	  timer_sub.Start();
    
    convertRgb2Gray(input_matrix, crop_width, height, gray_matrix);

    timer_sub.Stop();
    t1 += timer_sub.Elapsed();
    
    timer_sub.Start();
    
    generateEnergyMatrix(gray_matrix, crop_width, height, energy_matrix);

    timer_sub.Stop();
    t2 += timer_sub.Elapsed();

    timer_sub.Start();
    
    generateSeams(energy_matrix, crop_width, height, back_track_matrix, min_energy_matrix);

    timer_sub.Stop();
    t3 += timer_sub.Elapsed();

    timer_sub.Start();
    
    getSmallestSeam(min_energy_matrix, crop_width, height, seam, back_track_matrix);
    removeSeam(input_matrix, seam, crop_width, height);

    timer_sub.Stop();
    t4 += timer_sub.Elapsed();

		free(min_energy_matrix);
    free(back_track_matrix);
  	free(gray_matrix);
		free(energy_matrix);
    free(seam);

    crop_width = crop_width - 1;
  }

  memcpy(outPixels, input_matrix, scale_width * height * sizeof(uchar3));
  free(input_matrix);

  printf("\nConvert RGB to gray run time: %f ms\n", t1);
  printf("Gengerate energy matrix run time: %f ms\n", t2);
  printf("Generate all seams run time: %f ms\n", t3);
  printf("Find smallest seam and remove run time: %f ms\n", t4);
	
  timer_sum.Stop();
	printf("\nToal run time: %f ms\n", timer_sum.Elapsed());
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

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main(int argc, char ** argv)
{
	if (argc != 3 && argc != 4 && argc != 5)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}
    
	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nInput image size (width x height): %i x %i\n", width, height);
  float scale_rate = 0.85;

  if (argc >= 4) 
  {
      scale_rate = atof(argv[3]);
  }
  int scale_width = width * scale_rate;
  printf("Output image size (width x height): %i x %i\n", scale_width, height);

	uchar3 * outPixels = (uchar3 *)malloc(scale_width * height * sizeof(uchar3)); 
	
  if (argc == 5) {
    seamCarvingCalTime(inPixels, width, height, outPixels, scale_width);
  } else {
    seamCarving(inPixels, width, height, outPixels, scale_width);
  }
  

  //char * outFileNameBase = strtok(argv[2], "."); 
	//writePnm(outPixels, scale_width, height, concatStr(outFileNameBase, "_host.pnm"));

  writePnm(outPixels, scale_width, height, concatStr("output/", "out_host.pnm"));

	free(inPixels);
	free(outPixels);

}
