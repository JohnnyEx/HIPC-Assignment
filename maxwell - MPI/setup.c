#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>

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

	X = X / size;
}

/**
 * @brief Allocate all of the arrays used for computation
 * 
 */
void allocate_arrays() {
	Ex_size_x = X; Ex_size_y = Y+1;
	Ex = alloc_2d_array(Ex_size_x+1, Ex_size_y); // + 1 for ghosting
	Ey_size_x = X+1; Ey_size_y = Y;
	Ey = alloc_2d_array(Ey_size_x, Ey_size_y);
	
	Bz_size_x = X; Bz_size_y = Y;
	Bz = alloc_2d_array(Bz_size_x+1, Bz_size_y); // + 1 for ghosting 
	
	E_size_x = X+1; E_size_y = Y+1; E_size_z = 3;
	E = alloc_3d_array(E_size_x, E_size_y, E_size_z);

	B_size_x = X+1; B_size_y = Y+1; B_size_z = 3;
	B = alloc_3d_array(B_size_x, B_size_y, B_size_z);
	
	global_E = alloc_3d_array((E_size_x-1)*size+1, E_size_y, E_size_z);
	global_B = alloc_3d_array((B_size_x-1)*size+1, B_size_y, B_size_z);
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
	free_3d_array(global_E);
	free_3d_array(global_B);
}

/**
 * @brief Set up a guassian to curve around the centre
 * 
 */
void problem_set_up() {
	int ex_i = rank * Ex_size_x; // To take the absolute horizontal iteration
	int ey_i = rank * (Ey_size_x - 1);
	double xcen = lengthX / 2.0;
	double ycen = lengthY / 2.0;

    for (int i = ex_i; i < ex_i + Ex_size_x; i++ ) {
        for (int j = 0; j < Ex_size_y; j++) {
            double xcoord = (i - xcen) * dx;
            double ycoord = j * dy;
            double rx = xcen - xcoord;
            double ry = ycen - ycoord;
            double rlen = sqrt(rx*rx + ry*ry);
			double tx = (rlen == 0) ? 0 : ry / rlen;
            double mag = exp(-400.0 * (rlen - (lengthX / 4.0)) * (rlen - (lengthX / 4.0)));
            Ex[i-ex_i+1][j] = mag * tx;
		}
	}
    for (int i = ey_i; i < ey_i + Ey_size_x-1; i++ ) {
        for (int j = 0; j < Ey_size_y; j++) {
            double xcoord = i * dx;
            double ycoord = (j - ycen) * dy;
            double rx = xcen - xcoord;
            double ry = ycen - ycoord;
            double rlen = sqrt(rx*rx + ry*ry);
            double ty = (rlen == 0) ? 0 : -rx / rlen;
			double mag = exp(-400.0 * (rlen - (lengthY / 4.0)) * (rlen - (lengthY / 4.0)));
            Ey[i-ey_i][j] = mag * ty;
		}
	}
}