__kernel void heat_dissipation(__global float* a, __global float* b, int m, int n, int np, float td, float h)
{
  int id = get_global_id(0);

  int i = id / (n - 2) + 1;
  int j = id % (n - 2) + 1;

  int index = i * n + j;

  a[index] = (1.0 - 4*td / h*h) * b[index] +
    (td/h*h) * (b[index - n] +
                b[index + n] +
                b[index - 1] +
                b[index + 1]);
}
