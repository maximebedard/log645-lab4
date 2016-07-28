__kernel void heat_dissipation(__global float* a, __global float* b, int m, int n, int np, float td, float h)
{
  int index = get_global_id(0);

  a[index] = (1.0 - 4*td / h*h) * b[index] +
    (td/h*h) * (b[index - n] +
                b[index + n] +
                b[index - 1] +
                b[index + 1]);
}
