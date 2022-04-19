#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "vtk.h"
#include "data.h"
#include "setup.h"

/**
 * @brief Set up some default values before arguments have been loaded
 * 
 */
void set_defaults() {
	lengthX = 1.0;
	lengthY = 1.0;

	X = 4000;
	Y = 4000;
	
	T = 1.6e-9;

	set_default_base();
}

/**
 * @brief Set up some of the values required for computation after arguments have been loaded
 * 
 */
void setup() {
	dx = lengthX / X;
	dy = lengthY / Y;

	dt = cfl * (dx > dy ? dx : dy) / c;
	
	if (steps == 0) // only set this if steps hasn't been specified
		steps = (int) (T / dt);
}

/**
 * @brief Allocate all of the arrays used for computation
 * 
 */
void allocate_arrays(int rank, int size) {
	// Add ghost columns
	Ex_size_x = X; Ex_size_y = Y+1;
	Ex = alloc_2d_array(X + 1, Y + 1);
	Ey_size_x = X+1; Ey_size_y = Y;
	Ey = alloc_2d_array(X + 1, Y);
	
	Bz_size_x = X; Bz_size_y = Y;
	Bz = alloc_2d_array(X + 1, Y);
	
	E_size_x = X + 1; E_size_y = Y + 1; E_size_z = 3;
	E = alloc_3d_array(E_size_x, E_size_y, E_size_z);

	B_size_x = X + 1; B_size_y = Y + 1; B_size_z = 3;
	B = alloc_3d_array(B_size_x, B_size_y, B_size_z);
}

/**
 * @brief Free all of the arrays used for the computation
 * 
 */
void free_arrays() {
	free_2d_array(Ex);
	free_2d_array(Ey);
	free_2d_array(Bz);
	free_3d_array(E);
	free_3d_array(B);
}

/**
 * @brief Set up a guassian to curve around the centre
 * 
 */
void problem_set_up(int rank, int size) {
	int mystartEx = (rank == 0) ? 0 : 1;
    int myendEx = (rank == size-1) ? Ex_size_y : Ex_size_y-1;
    for (int i = 0; i < Ex_size_x; i++ ) {
        for (int j = mystartEx; j < myendEx; j++) {
            double xcen = lengthX / 2.0;
            double ycen = lengthY / 2.0;
            double xcoord = (i - xcen) * dx;
            double ycoord = j * dy;
            double rx = xcen - xcoord;
            double ry = ycen - ycoord;
            double rlen = sqrt(rx*rx + ry*ry);
			double tx = (rlen == 0) ? 0 : ry / rlen;
            double mag = exp(-400.0 * (rlen - (lengthX / 4.0)) * (rlen - (lengthX / 4.0)));
            Ex[i+1][j] = mag * tx;
		}
	}
	int mystart = (rank == 0) ? 0 : 1;
    int myend = (rank == size-1) ? Ey_size_y : Ey_size_y-1;
    for (int i = 0; i < Ey_size_x; i++ ) {
        for (int j = mystart; j < myend; j++) {
            double xcen = lengthX / 2.0;
            double ycen = lengthY / 2.0;
            double xcoord = i * dx;
            double ycoord = (j - ycen) * dy;
            double rx = xcen - xcoord;
            double ry = ycen - ycoord;
            double rlen = sqrt(rx*rx + ry*ry);
            double ty = (rlen == 0) ? 0 : -rx / rlen;
			double mag = exp(-400.0 * (rlen - (lengthY / 4.0)) * (rlen - (lengthY / 4.0)));
            Ey[i][j] = mag * ty;
		}
	}
}
