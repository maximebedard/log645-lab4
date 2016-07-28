#include <assert.h>
#include <string.h>
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
  cl_program program             = NULL;
  cl_kernel kernel               = NULL;
  cl_platform_id platform_id     = NULL;

  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  char string[MEM_SIZE];

  int mem_size_A = sizeof(float) * m * n;
  float* h_matrix = (float*) malloc(mem_size_A);
  float* j_matrix = (float*) malloc(mem_size_A);

  // yolo refactor this pls.
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      h_matrix[i * n + j] = matrix[0][i][j];
    }
  }

  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      j_matrix[i * m + j] = matrix[1][i][j];
    }
  }

  cl_mem d_matrix, e_matrix;

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
  d_matrix = clCreateBuffer(
    context,
    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
    mem_size_A,
    h_matrix,
    &ret
  );

  e_matrix = clCreateBuffer(
    context,
    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
    mem_size_A,
    j_matrix,
    &ret
  );

  if (!d_matrix || !e_matrix)
  {
    printf("Error: Failed to allocate device memory!\n");
    exit(1);
  }

  // Create Kernel Program from the source
  program = clCreateProgramWithSource(
    context,
    1,
    (const char **)&source_str,
    (const size_t *)&source_size,
    &ret
  );

  // Build Kernel Program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];
    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
    buffer, &len);

    printf("%s\n", buffer);
    exit(1);
  }

  // Create OpenCL Kernel
  kernel = clCreateKernel(program, "heat_dissipation", &ret);

  // set workgroups/workitems
  size_t global_item_size = (m - 2) * (n - 2); // Process the entire lists
  size_t local_item_size = 8; // Process in groups of 8

  // Set OpenCL Kernel Parameters
  ret =  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_matrix);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&e_matrix);
  ret |= clSetKernelArg(kernel, 2, sizeof(int), (void *)&m);
  ret |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&n);
  ret |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&np);
  ret |= clSetKernelArg(kernel, 5, sizeof(float), (void *)&td);
  ret |= clSetKernelArg(kernel, 6, sizeof(float), (void *)&h);

  if (ret != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", ret);
    exit(1);
  }
  int current = 0;
  for(int k = 0; k < np; k++) {
    if(current == 0) {
      ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_matrix);
      ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&e_matrix);
    } else {
      ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&e_matrix);
      ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_matrix);
    }

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
      &global_item_size, NULL, 0, NULL, NULL);

    current = k % 2;
  }

  // Copy results from the memory buffer
  ret = clEnqueueReadBuffer(
    command_queue,
    (current == 0 ? d_matrix : e_matrix),
    CL_TRUE,
    0,
    mem_size_A,
    &matrix[0],
    0,
    NULL,
    NULL
  );

  // Finalization
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(d_matrix);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(h_matrix);
  free(j_matrix);
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
