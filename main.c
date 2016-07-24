#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#define DEBUG 1

#ifdef DEBUG
#  define DEBUG_PRINT(...) do{ fprintf( stderr, __VA_ARGS__ ); } while(0)
#else
#  define DEBUG_PRINT(...) do{ } while (0)
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

void heat_dissipation_seq(int m, int n, float matrix[2][m][n], int np, float td, float h);
void heat_dissipation_par(int m, int n, float matrix[2][m][n], int np, float td, float h);

void matrix_init(int m, int n, float matrix[2][m][n]);
void matrix_print(int m, int n, float matrix[2][m][n]);

double get_current_time();

int main(int argc, char const *argv[]) {
  int m    = 0;
  int n    = 0;
  int np   = 0;
  float td = 0.0f;
  float h  = 0.0f;
  double start, end;

  m    = atoi(argv[1]);
  n    = atoi(argv[2]);
  np   = atoi(argv[3]);
  td   = atof(argv[4]);
  h    = atof(argv[5]);

  float matrix[2][m][n];

  printf("m=%d, n=%d, np=%d, td=%.5f, h=%.5f\n", m, n, np, td, h);

  matrix_init(m, n, matrix);
  DEBUG_PRINT("seq-init\n========\n");
  matrix_print(m, n, matrix);
  start = get_current_time();
  heat_dissipation_seq(m, n, matrix, np, td, h);
  end = get_current_time();
  double dt_seq = end - start;
  DEBUG_PRINT("seq-final\n=========\n");
  matrix_print(m, n, matrix);

  matrix_init(m, n, matrix);
  DEBUG_PRINT("par-init\n========\n");
  matrix_print(m, n, matrix);
  start = get_current_time();
  heat_dissipation_par(m, n, matrix, np, td, h);
  end = get_current_time();
  double dt_par = end - start;
  DEBUG_PRINT("par-final\n=========\n");
  matrix_print(m, n, matrix);

  printf("|seq|par|acceleration|\n");
  printf("|---|---|------------|\n");
  printf("|%.8f|%.8f|%.8f|\n", dt_seq, dt_par, dt_seq / dt_par);

  return 0;
}

void heat_dissipation_seq(int m, int n, float matrix[2][m][n], int np, float td, float h) {
  int i, j, k;
  int current = 0;

  for(k = 1; k < np; k++) {
    for(i = 1; i < m - 1; i++) {
      for(j = 1; j < n - 1; j++) {
        matrix[current][i][j] = (1.0 - 4*td / h*h) * matrix[1-current][i][j] +
          (td/h*h) * (matrix[1-current][i - 1][j] +
                  matrix[1-current][i + 1][j] +
                  matrix[1-current][i][j - 1] +
                  matrix[1-current][i][j + 1]);
      }
    }

    current = k % 2;
  }
}

void heat_dissipation_par(int m, int n, float matrix[2][m][n], int np, float td, float h) {
  cl_device_id device_id         = NULL;
  cl_context context             = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobj                  = NULL;
  cl_program program             = NULL;
  cl_kernel kernel               = NULL;
  cl_platform_id platform_id     = NULL;

  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  // memoire pour les matrices utilisés
  cl_mem d_u0;
  cl_mem d_u1;

  char string[MEM_SIZE];

  FILE *fp;
  char *source_str;
  size_t source_size;

  // Load the source code containing the kernel
  fp = fopen("./main.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get Platform and Device Info
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  // Create Command Queue
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  // Create Memory Buffer
  memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(char), NULL, &ret);

  // Create Kernel Program from the source
  program = clCreateProgramWithSource(
    context,
    1,
    (const char **)&source_str,
    (const size_t *)&source_size, &ret
  );

  // Build Kernel Program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  // Create OpenCL Kernel
  kernel = clCreateKernel(program, "heat_dissipation", &ret);

  // Set OpenCL Kernel Parameters
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);

  // Execute OpenCL Kernel
  ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);

  // Copy results from the memory buffer
  ret = clEnqueueReadBuffer(
    command_queue, memobj, CL_TRUE, 0,
    MEM_SIZE * sizeof(char),string, 0, NULL, NULL
  );

  // Display Result
  puts(string);

  // Finalization
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(source_str);
}

double get_current_time() {
  struct timeval tp;
  gettimeofday (&tp, NULL);
  return (double) (tp.tv_sec) + (double) (tp.tv_usec) / 1e6;
}

void matrix_init(int m, int n, float matrix[2][m][n]) {
  int i, j;

  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      matrix[0][i][j] = (double)(i * (m - i - 1)) * (j * (n - j - 1));
      matrix[1][i][j] = matrix[0][i][j];
    }
  }
}

void matrix_zero(int m, int n, float matrix[2][m][n]) {
  int i, j;

  for(i = 0; i < m ; i++){
    for(j = 0; j < n; j++){
      matrix[0][i][j] = 0.0;
      matrix[1][i][j] = matrix[0][i][j];
    }
  }
}

void matrix_print(int m, int n, float matrix[2][m][n]) {
  int i, j;
  for(i = 0; i < m; i++) {
    for(j = 0; j < n; j++) {
      DEBUG_PRINT("|%15.2f", matrix[0][i][j]);
    }
    DEBUG_PRINT("|\n");
  }
}
