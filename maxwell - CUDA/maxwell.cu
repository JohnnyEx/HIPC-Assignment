#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "args.h"
#include "vtk.h"
#include "data.h"
#include "setup.h"

/**
 * @brief Update the magnetic and electric fields. The magnetic fields are updated for a half-time-step. The electric fields are updated for a full time-step.
 * 
 */
__global__ void update_fields() {
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int cx_ex = Ex_size_x / (gridDim.x * blockDim.x); // Cells per cuda thread in X direction (keeping this same for all fields)
    int cy_ey = Ey_size_y / (gridDim.y * blockDim.y); // Cells per cuda thread in Y direction

    for (int i = cx_ex * tidx; i < cx_ex * (tidx + 1); i++) {
        for (int j = cy_ey * tidy; j < cy_ey * (tidy + 1); j++) {
            Bz[i * bz_pitch + j] = Bz[i * bz_pitch + j] - (dt / dx) * (Ey[(i+1) * ey_pitch + j] - Ey[i * ey_pitch + j])
                                                 + (dt / dy) * (Ex[i * ex_pitch + j + 1] - Ex[i * ex_pitch + j]);
        }
    }

    for (int i = cx_ex * tidx; i < cx_ex * (tidx + 1); i++) {
        for (int j = cy_ey * tidy; j < cy_ey * (tidy + 1); j++) {
            if (tidy == 0 && j == 0)
                continue;
            Ex[i * ex_pitch + j] = Ex[i * ex_pitch + j] + (dt / (dy * constants.eps * constants.mu)) * (Bz[i * bz_pitch + j] - Bz[i * bz_pitch + j - 1]);
        }
    }

    for (int i = cx_ex * tidx; i < cx_ex * (tidx + 1); i++) {
        for (int j = cy_ey * tidy; j < cy_ey * (tidy + 1); j++) {
            if (tidx == 0 && i == 0)
                continue;
            Ey[i * ey_pitch + j] = Ey[i * ey_pitch + j] - (dt / (dx * constants.eps * constants.mu)) * (Bz[i * bz_pitch + j] - Bz[(i - 1) * bz_pitch + j]);
        }
    }
}

/**
 * @brief Apply boundary conditions
 * 
 */
__global__ void apply_boundary() {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int cx_ex = Ex_size_x / (gridDim.x * blockDim.x); // Cells per cuda thread in X direction
    int cy_ey = Ey_size_y / (gridDim.y * blockDim.y); // Cells per cuda thread in Y direction

    for (int i = tidx * cx_ex; i < cx_ex * (tidx + 1); i++) {
        if (tidy == 0)
            Ex[i * ex_pitch] = -Ex[i * ex_pitch + 1];
        if (tidy == gridDim.y * blockDim.y - 1)
            Ex[i * ex_pitch + Ex_size_y - 1] = -Ex[i * ex_pitch + Ex_size_y - 2];
    }

    for (int j = tidy * cy_ey; j < cy_ey * (tidy + 1); j++) {
        if (tidx == 0)
            Ey[j] = -Ey[ey_pitch + j];
        if (tidx == gridDim.x * blockDim.x - 1)
            Ey[ey_pitch * (Ey_size_x - 1) + j] = -Ey[ey_pitch * (Ey_size_x - 2) + j];
    }
}

/**
 * @brief Resolve the Ex, Ey and Bz fields to grid points and sum the magnitudes for output
 * 
 * @param E_mag The returned total magnitude of the Electric field (E)
 * @param B_mag The returned total magnitude of the Magnetic field (B) 
 */
__global__ void resolve_to_grid(double *E_mag, double *B_mag) {
	*E_mag = 0.0;
	*B_mag = 0.0;
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int cx_x = (E_size_x - 1) / (gridDim.x * blockDim.x); // Cells per cuda thread in X direction
    int cy_y = (E_size_y - 1) / (gridDim.y * blockDim.y); // Cells per cuda thread in Y direction
    E_mag[tidy * gridDim.x * blockDim.x + tidx] = 0.0;
    B_mag[tidy * gridDim.x * blockDim.x + tidx] = 0.0;

    for (int i = cx_x * tidx; i < cx_x * (tidx + 1); i++) {
        for (int j = cy_y * tidy; j < cy_y * (tidy + 1); j++) {
            if ((tidx == 0 && i == 0) || (tidy == 0 && j == 0))
                continue;
            E[i * e_pitch + j * E_size_z] = (Ex[(i-1) * ex_pitch + j] + Ex[i * ex_pitch + j]) / 2.0;
            E[i * e_pitch + j * E_size_z + 1] = (Ey[i * ey_pitch + j - 1] + Ey[i * ey_pitch + j]) / 2.0;

            E_mag[tidy * gridDim.x * blockDim.x + tidx] += sqrt((E[i * e_pitch + j * E_size_z] * E[i * e_pitch + j * E_size_z])
                                                                + (E[i * e_pitch + j * E_size_z + 1] * E[i * e_pitch + j * E_size_z + 1]));
        }
    }

    for (int i = cx_x * tidx; i < cx_x * (tidx + 1); i++) {
        for (int j = cy_y * tidy; j < cy_y * (tidy + 1); j++) {
            if ((tidx == 0 && i == 0) || (tidy == 0 && j == 0))
                continue;
            B[i * b_pitch + j * B_size_z + 2] = (Bz[(i-1) * bz_pitch + j] + Bz[i * bz_pitch + j]
                                                                      + Bz[i * bz_pitch + j - 1] + Bz[(i-1) * bz_pitch + j - 1]) / 4.0;

            B_mag[tidy * gridDim.x * blockDim.x + tidx] += sqrt(B[i * b_pitch + j * B_size_z + 2] * B[i * b_pitch + j * B_size_z + 2]);
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
	set_defaults();
	parse_args(argc, argv);
	setup();

	printf("Running problem size %f x %f on a %d x %d grid.\n", lengthX, lengthY, X, Y);
	
	if (verbose) print_opts();
	
	allocate_arrays();

    // Cuda setup
    dim3 block = dim3(block_x, block_y);
    dim3 grid = dim3(grid_x, grid_y);
    int noThreads = grid.x * block.x * grid.y  * block.y;
	problem_set_up<<<1,1>>>();
    double *E_mag_cudaV = (double *) calloc(noThreads, sizeof(double));
    double *B_mag_cudaV = (double *) calloc(noThreads, sizeof(double));
	double *d_E_mag_cudaV, *d_B_mag_cudaV;
    cudaMalloc(&d_E_mag_cudaV, noThreads * sizeof(double));
    cudaMalloc(&d_B_mag_cudaV, noThreads * sizeof(double));
    long e_pitch_host = E_size_y * E_size_z * sizeof(double);
    long b_pitch_host = B_size_y * B_size_z * sizeof(double);


    // start at time 0
	double t = 0.0;
	int i = 0;
	while (i < steps) {
		apply_boundary<<<grid, block>>>();
		update_fields<<<grid, block>>>();

		t += dt;

		if (i % output_freq == 0) {
			double E_mag = 0, B_mag = 0;
			resolve_to_grid<<<grid, block>>>(d_E_mag_cudaV, d_B_mag_cudaV);
            cudaMemcy(E_mag_cudaV, d_E_mag_cudaV, total_threads * sizeof(double), cudaMemcyDeviceToHost);
            cudaMemcy(B_mag_cudaV, d_B_mag_cudaV, total_threads * sizeof(double), cudaMemcyDeviceToHost);
            for (int j = 0; j < noThreads; j++){
                E_mag += E_mag_cudaV[j];
                B_mag += B_mag_cudaV[j];
            }

			printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);

			if ((!no_output) && (enable_checkpoints))
            {
                cudaMemcy2D(&host_E[0][0][0], e_pitch_host, E, e_pitch * sizeof(double), e_pitch_host, E_size_x, cudaMemcyDeviceToHost);
                cudaMemcy2D(&host_B[0][0][0], b_pitch_host, B, b_pitch * sizeof(double), b_pitch_host, B_size_x, cudaMemcyDeviceToHost);
                write_checkpoint(i);
            }

		}

		i++;
	}

	double E_mag, B_mag;
	resolve_to_grid<<<grid, block>>>(&E_mag, &B_mag);
    cudaMemcy(E_mag_cudaV, d_E_mag_cudaV, noThreads * sizeof(double), cudaMemcyDeviceToHost);
    cudaMemcy(B_mag_cudaV, d_B_mag_cudaV, noThreads * sizeof(double), cudaMemcyDeviceToHost);
    for (int i = 0; i < noThreads; i++){
        E_mag += E_mag_cudaV[i];
        B_mag += B_mag_cudaV[i];
    }

	printf("Step %8d, Time: %14.8e (dt: %14.8e), E magnitude: %14.8e, B magnitude: %14.8e\n", i, t, dt, E_mag, B_mag);
	printf("Simulation complete.\n");

	if (!no_output) {
        cudaMemcy2D(&host_E[0][0][0], e_pitch_host, E, e_pitch * sizeof(double), e_pitch_host, E_size_x, cudaMemcyDeviceToHost);
        cudaMemcy2D(&host_B[0][0][0], b_pitch_host, B, b_pitch * sizeof(double), b_pitch_host, B_size_x, cudaMemcyDeviceToHost);
        write_result();
    }
    
	free_arrays();

	exit(0);
}


