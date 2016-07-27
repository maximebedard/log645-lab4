__kernel void heat_dissipation(
  __global float* matrix,
  int m,
  int n,
  int np,
  float td,
  float h)
{
  // int index = x + y * sizeX + z * sizeX * sizeY;
  printf("matrix=%.2f, m=%d, n=%d np=%d, td=%.f, h=%.f\n", &matrix[15], m, n, np, td, h);
}
