#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "vtk.h"
#include "data.h"
#include "setup.h"
#include "args.h"

char checkpoint_basename[1024];
char result_filename[1024];

/**
 * @brief Set the default basename for file output to out/maxwell
 *
 */
void set_default_base() {
    set_basename("maxwell");
}

/**
 * @brief Set the basename for file output
 *
 * @param base Basename string
 */
void set_basename(char *base) {
    checkpoint_basename[0] = '\0';
    result_filename[0] = '\0';
    sprintf(checkpoint_basename, "%s-%%d.vtk", base);
    sprintf(result_filename, "%s.vtk", base);
}

/**
 * @brief Get the basename for file output
 *
 * @return char* Basename string
 */
char *get_basename() {
    return checkpoint_basename;
}

/**
 * @brief Write a checkpoint VTK file (with the iteration number in the filename)
 *
 * @param iteration The current iteration number
 * @return int Return whether the write was successful
 */
int write_checkpoint(int iteration) {
    char filename[1024];
    char comp_filename[1024];
    sprintf(filename, checkpoint_basename, iteration);
    return write_vtk(filename, comp_filename);
}

/**
 * @brief Write the final output to a VTK file
 *
 * @return int Return whether the write was successful
 */
int write_result() {
    return write_vtk(result_filename, NULL);
}

/**
 * @brief Write a VTK file with the current state of the Electric and Magnetic Fields
 *
 * @param filename The filename to write out
 * @return int Return whether the write was successful
 */
int write_vtk(char* filename, char *comp_filename) {
    FILE * f;
    f = fopen(filename, "w");
    if (f == NULL) {
        perror("Error");
        return -1;
    }
	char *buffer;
	size_t bufsize = 1024;
    int comp_line_len = 0;

    // Adjusting for local array sizes
    int realX = X * size;
    int realY = Y;

        // Write the VTK header information
        fprintf(f, "# vtk DataFile Version 3.0\n");
        fprintf(f, "Karman Output\n");
        fprintf(f, "ASCII\n");
        fprintf(f, "DATASET RECTILINEAR_GRID\n");

        // Write out the grid information
        fprintf(f, "DIMENSIONS %d %d 1\n", (realX+1), (realY+1));
        fprintf(f, "X_COORDINATES %d float\n", (realX+1));
        for (int i = 0; i <= realX; i++) fprintf(f, "   %.12e", (lengthX * ((double) i / (realX+1))));
        fprintf(f, "\nY_COORDINATES %d float\n", (realY+1));
        for (int i = 0; i <= realY; i++) fprintf(f, "   %.12e", (lengthY * ((double) i / (realY+1))));
        fprintf(f, "\nZ_COORDINATES 1 float\n");
        fprintf(f, "  0.000000000000e+00");

        fprintf(f, "\nPOINT_DATA %d\n", ((realX+1) * (realY+1)));

        // Write out the E and B vector fields
        fprintf(f, "VECTORS E_field float\n");
    
    for (int j = 0; j <= realY; j++) {
        for (int i = 0; i <= realX; i++) {
            fprintf(f, "  %.12e %.12e 0.000000000000e+00\n", global_E[i][j][0], global_E[i][j][1]);
        
    }

    fprintf(f, "VECTORS B_field float\n");
    for (int j = 0; j <= realY; j++) {
        for (int i = 0; i <= realX; i++)
                fprintf(f, "  0.000000000000e+00 0.000000000000e+00 %.12e\n", global_B[i][j][2]);
    }

    fclose(f);
    return 0;
}
