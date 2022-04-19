#ifndef DATA_H
#define DATA_H

extern const double c; // Speed of light
extern const double mu; // permiability of free space
extern const double eps; // permitivitty of free space

// must be less than 1/sqrt(2) in 2D
extern const double cfl;

// Grid size in metres
extern double lengthX;
extern double lengthY;

// Discretisation in cells
extern int X;
extern int Y;

// dx, dy, dt constants.
extern double dx;
extern double dy;
extern double dt;

// Time to run for / or number of steps
extern double T;
extern int steps;

// Cuda
extern int grid_x;
extern int grid_y;
extern int block_x;
extern int block_y;

// x = Ex values
// o = Ey values
// * = Bz values
// + = E and B values
//
//    +---x---+---x---+---x---+---x---+---x---+
//    |       |       |       |       |       |
//    o   *   o   *   o   *   o   *   o   *   o
//    |       |       |       |       |       |
//    +---x---+---x---+---x---+---x---+---x---+
//    |       |       |       |       |       |
//    o   *   o   *   o   *   o   *   o   *   o
//    |       |       |       |       |       |
//    +---x---+---x---+---x---+---x---+---x---+
// ^  |       |       |       |       |       |
// |  o   *   o   *   o   *   o   *   o   *   o
// y  |       |       |       |       |       |
//    +---x---+---x---+---x---+---x---+---x---+
//(0,0)  x -> 

extern int Ex_size_x, Ex_size_y;
extern double ** Ex;
extern size_t ex_pitch;
extern int Ey_size_x, Ey_size_y;
extern double ** Ey;
extern size_t ey_pitch;
extern int Bz_size_x, Bz_size_y;
extern double ** Bz;
extern size_t bz_pitch;

// These array are only needed for visualisation
extern int E_size_x, E_size_y, E_size_z;
extern double *** E;
extern size_t e_pitch;
extern int B_size_x, B_size_y, B_size_z;
extern double *** B;
extern size_t b_pitch;

extern double *** host_E;
extern double *** host_B;

void alloc_2d_array(int m, int n, double **array, size_t *pitch);
void free_2d_array(double ** array);
void alloc_3d_cuda_array(int m, int n, int o, size_t *pitch);
void free_3d_cuda_array(double *array);
double ***alloc_3d_array(int m, int n, int o);
void free_3d_array(double*** array);

#endif