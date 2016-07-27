__kernel void heat_dissipation(__global float* matrix, int m, int n, int np, float td, float h)
{
  // int index = x + y * sizeX + z * sizeX * sizeY;
  int tx = get_global_id(0);
  int ty = get_global_id(1);
  printf("global id : %d %d", tx, ty);
  //printf("matrix=%.f, m=%d, n=%d np=%d, td=%.f, h=%.f\n", matrix[12], m, n, np, td, h);
}
