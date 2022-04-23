#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include <string.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
void update_fields(MPI_Datatype datatype1, MPI_Datatype datatype2, MPI_Datatype datatype3) {

	// halo exchange for Ey array
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Sendrecv(Ey[0], 1, datatype2, left, 13, Ey[Ey_size_x-1], 1, datatype2, right, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	for (int i = 1; i < Bz_size_x+1; i++) {
		for (int j = 0; j < Bz_size_y; j++) {
			Bz[i][j] = Bz[i][j] - (dt / dx) * (Ey[i][j] - Ey[i-1][j])
				                + (dt / dy) * (Ex[i][j+1] - Ex[i][j]);
		}
	}

	for (int i = 1; i < Ex_size_x+1; i++) {
		for (int j = 1; j < Ex_size_y-1; j++) {
			Ex[i][j] = Ex[i][j] + (dt / (dy * eps * mu)) * (Bz[i][j] - Bz[i][j-1]);
		}
	}

	// halo exchange for Bz array
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Sendrecv(Bz[Bz_size_x], 1, datatype3, right, 13, Bz[0], 1, datatype3, left, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	for (int i = 0; i < Ey_size_x-1; i++) {
		for (int j = 0; j < Ey_size_y; j++) {
			Ey[i][j] = Ey[i][j] - (dt / (dx * eps * mu)) * (Bz[i+1][j] - Bz[i][j]);
		}
	}

	// halo exchange for Ex array
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Sendrecv(Ex[Ex_size_x], 1, datatype1, right, 13, Ex[0], 1, datatype1, left, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/**
 * @brief Apply boundary conditions
 * 
 */
void apply_boundary() {
	for (int i = 1; i < Ex_size_x+1; i++) {
		Ex[i][0] = -Ex[i][1];
		Ex[i][Ex_size_y-1] = -Ex[i][Ex_size_y-2];
	}

	// condition imposed to check rank 0
	for (int j = 0; j < Ey_size_y; j++) {
		if (rank == 0)
			Ey[0][j] = -Ey[1][j];
		if (rank == size -1)
			Ey[Ey_size_x-1][j] = -Ey[Ey_size_x-2][j];
	}
}

/**
 * @brief Resolve the Ex, Ey and Bz fields to grid points and sum the magnitudes for output
 * 
 * @param E_mag The returned total magnitude of the Electric field (E)
 * @param B_mag The returned total magnitude of the Magnetic field (B) 
 */
void resolve_to_grid(double *E_mag, double *B_mag) {
	*E_mag = 0.0;
	*B_mag = 0.0;
	
	// control parameters adjusted to check rank 0
	for (int i = rank == 0 ? 1 : 0; i < E_size_x-1; i++) {
		for (int j = 1; j < E_size_y-1; j++) {
			E[i][j][0] = (Ex[i][j] + Ex[i+1][j]) / 2.0;
			E[i][j][1] = (Ey[i][j-1] + Ey[i][j]) / 2.0;

			*E_mag += sqrt((E[i][j][0] * E[i][j][0]) + (E[i][j][1] * E[i][j][1]));
		}
	}

	// control parameters adjusted to check rank 0
	for (int i = rank == 0 ? 1 : 0; i < B_size_x-1; i++) {
		for (int j = 1; j < B_size_y-1; j++) {
			B[i][j][2] = (Bz[i][j] + Bz[i+1][j] + Bz[i+1][j-1] + Bz[i][j-1]) / 4.0;

			*B_mag += sqrt(B[i][j][2] * B[i][j][2]);
		}
	}
}

/**
 * @brief The main routine that sets up the problem and executes the timestepping routines
 * 
 * @param argc The number of arguments passed to the program
 * @param argv An array of the arguments passed to the program
 * @return int The return value of the application
 */
int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	set_defaults();
	parse_args(argc, argv);
	setup();
	total_error = 0;

	/**
	 * @brief
	 * setting up for comparison
	 */
	FILE *comp_file;
	char comp_file_name[1024];
	char *buffer;
	size_t bufsize = 1024;
	int comp_line_len = 0;
	if (rank == 0) {
		if (comp_mode != 0) {
			sprintf(comp_file_name_base, "comp/comp_%d_%d%%s", X * size, Y);
			if (comp_mode == 1) {
				sprintf(comp_file_name, comp_file_name_base, "_mag.cmp");
				comp_file = fopen(comp_file_name, "r");
				if (comp_file == NULL) {
					printf("Unable to find comparison file %s\n", comp_file_name);
					exit(1);
				}
			}
		}
		printf("Running problem size %f x %f on a %d x %d grid.\n", lengthX, lengthY, X*size, Y);
	}
	
	if (verbose) print_opts();
	
	allocate_arrays();

	problem_set_up();

	// start at time 0
	double t = 0.0;
	int i = 0;
	double global_E_mag, global_B_mag;

	/**
	 * @brief 
	 * setting up MPI datatypes for 2D arrays to use in halo exchanges
	 */
	MPI_Datatype ex_column, ey_column, bz_column;
	MPI_Type_vector(1, Ex_size_y, Ex_size_y, MPI_DOUBLE, &ex_column);
	MPI_Type_commit(&ex_column);
	MPI_Type_vector(1, Ey_size_y, Ey_size_y, MPI_DOUBLE, &ey_column);
	MPI_Type_commit(&ey_column);
	MPI_Type_vector(1, Bz_size_y, Bz_size_y, MPI_DOUBLE, &bz_column);
	MPI_Type_commit(&bz_column);
	
	// calculate left and right ranks
	left = rank-1 < 0 ? MPI_PROC_NULL : rank-1;
	right = rank+1 >= size ? MPI_PROC_NULL: rank+1;
	
	/**
	 * @brief 
	 * setting up MPI datatype for 3D array to gather resolved data from all nodes
	 */
	MPI_Datatype global_3d_grid;
	MPI_Type_vector(E_size_x-1, E_size_z*E_size_y, E_size_z*E_size_y, MPI_DOUBLE, &global_3d_grid);
	MPI_Type_commit(&global_3d_grid);

	double start, end;
	start = MPI_Wtime();

	while (i < steps) {
		apply_boundary();
		update_fields(ex_column, ey_column, bz_column);
		t += dt;

		// reading lines here (regardless of output freq.) to keep track of lines
		if (comp_mode == 1 && rank == 0) {
			buffer = (char *) calloc(1024, sizeof(char));
			comp_line_len = getline(&buffer, &bufsize, comp_file);
		}

		if (i % output_freq == 0) {
			double E_mag, B_mag;
			resolve_to_grid(&E_mag, &B_mag);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Reduce(&E_mag, &global_E_mag, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&B_mag, &global_B_mag, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			if (rank == 0) {
				if (comp_mode == 1) {
					double mags[2] = { global_E_mag, global_B_mag };
					compare_line(comp_line_len, &buffer, mags);
				}
				printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, global_E_mag, global_B_mag);
			}
			
			if (enable_checkpoints && !no_output) {
				MPI_Gather(E[0][0], 1, global_3d_grid, global_E[0][0], 1, global_3d_grid, 0, MPI_COMM_WORLD);
				MPI_Gather(B[0][0], 1, global_3d_grid, global_B[0][0], 1, global_3d_grid, 0, MPI_COMM_WORLD);
			}
			
			if ((!no_output) && (enable_checkpoints) && rank == 0) {
				write_checkpoint(i);
			}
		}
		i++;
	}

	double E_mag, B_mag;
	resolve_to_grid(&E_mag, &B_mag);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&E_mag, &global_E_mag, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&B_mag, &global_B_mag, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, global_E_mag, global_B_mag);
		printf("Simulation complete.\n");
	}

	MPI_Type_free(&ey_column);
	MPI_Gather(E[0][0], 1, global_3d_grid, global_E[0][0], 1, global_3d_grid, 0, MPI_COMM_WORLD);
	MPI_Gather(B[0][0], 1, global_3d_grid, global_B[0][0], 1, global_3d_grid, 0, MPI_COMM_WORLD);
	
	end = MPI_Wtime();

	if (rank == 0)
		printf("Total elapsed time is %fs\n", end - start);
		
	if (!no_output && rank == 0)
		write_result();

	if (comp_mode != 0 && rank == 0)
		printf("Total error is %.15e\n", total_error);

	free_arrays();
	
	MPI_Finalize();

	exit(0);
}


