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
__global__ void update_fields(constants m_constants, arrays m_arrays, variables m_variables) {
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int cx_ex = m_arrays.Ex_size_x / (gridDim.x * blockDim.x); // Cells per cuda thread in X direction
    int cy_ey = m_arrays.Ey_size_y / (gridDim.y * blockDim.y); // Cells per cuda thread in Y direction

    for (int i = cx_ex * tidx; i < cx_ex * (tidx + 1); i++) {
        for (int j = cy_ey * tidy; j < cy_ey * (tidy + 1); j++) {
            m_arrays.Bz[i * m_arrays.bz_pitch + j] = m_arrays.Bz[i * m_arrays.bz_pitch + j] - (m_variables.dt / m_variables.dx) * (m_arrays.Ey[(i+1) * m_arrays.ey_pitch + j] - m_arrays.Ey[i * m_arrays.ey_pitch + j]) + (m_variables.dt / m_variables.dy) * (m_arrays.Ex[i * m_arrays.ex_pitch + j + 1] - m_arrays.Ex[i * m_arrays.ex_pitch + j]);
        }
    }

    for (int i = cx_ex * tidx; i < cx_ex * (tidx + 1); i++) {
        for (int j = cy_ey * tidy; j < cy_ey * (tidy + 1); j++) {
            if (tidy == 0 && j == 0)
                continue;
            m_arrays.Ex[i * m_arrays.ex_pitch + j] = m_arrays.Ex[i * m_arrays.ex_pitch + j] + (m_variables.dt / (m_variables.dy * m_constants.eps * m_constants.mu)) * (m_arrays.Bz[i * m_arrays.bz_pitch + j] - m_arrays.Bz[i * m_arrays.bz_pitch + j - 1]);
        }
    }

    for (int i = cx_ex * tidx; i < cx_ex * (tidx + 1); i++) {
        for (int j = cy_ey * tidy; j < cy_ey * (tidy + 1); j++) {
            if (tidx == 0 && i == 0)
                continue;
            m_arrays.Ey[i * m_arrays.ey_pitch + j] = m_arrays.Ey[i * m_arrays.ey_pitch + j] - (m_variables.dt / (m_variables.dx * m_constants.eps * m_constants.mu)) * (m_arrays.Bz[i * m_arrays.bz_pitch + j] - m_arrays.Bz[(i - 1) * m_arrays.bz_pitch + j]);
        }
    }
}

/**
 * @brief Apply boundary conditions
 * 
 */
__global__ void apply_boundary(arrays m_arrays) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int cx_ex = m_arrays.Ex_size_x / (gridDim.x * blockDim.x); // Cells per cuda thread in X direction
    int cy_ey = m_arrays.Ey_size_y / (gridDim.y * blockDim.y); // Cells per cuda thread in Y direction

    for (int i = tidx * cpx_ex; i < cpx_ex * (tidx + 1); i++) {
        if (tidy == 0)
            m_arrays.Ex[i * m_arrays.ex_pitch] = -m_arrays.Ex[i * m_arrays.ex_pitch + 1];
        if (tidy == gridDim.y * blockDim.y - 1)
            m_arrays.Ex[i * m_arrays.ex_pitch + m_arrays.Ex_size_y - 1] = -m_arrays.Ex[i * m_arrays.ex_pitch + m_arrays.Ex_size_y - 2];
    }

    for (int j = tidy * cpy_ey; j < cpy_ey * (tidy + 1); j++) {
        if (tidx == 0)
            m_arrays.Ey[j] = -m_arrays.Ey[m_arrays.ey_pitch + j];
        if (tidx == gridDim.x * blockDim.x - 1)
            m_arrays.Ey[m_arrays.ey_pitch * (m_arrays.Ey_size_x - 1) + j] = -m_arrays.Ey[m_arrays.ey_pitch * (m_arrays.Ey_size_x - 2) + j];
    }
}

/**
 * @brief Resolve the Ex, Ey and Bz fields to grid points and sum the magnitudes for output
 * 
 * @param E_mag The returned total magnitude of the Electric field (E)
 * @param B_mag The returned total magnitude of the Magnetic field (B) 
 */
__global__ void resolve_to_grid(double *E_mag, double *B_mag, arrays m_arrays) {
	*E_mag = 0.0;
	*B_mag = 0.0;
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int cx_x = (m_arrays.E_size_x - 1) / (gridDim.x * blockDim.x); // Cells per cuda thread in X direction
    int cy_y = (m_arrays.E_size_y - 1) / (gridDim.y * blockDim.y); // Cells per cuda thread in Y direction
    E_mag[tidy * gridDim.x * blockDim.x + tidx] = 0.0;
    B_mag[tidy * gridDim.x * blockDim.x + tidx] = 0.0;

    for (int i = cx_x * tidx; i < cx_x * (tidx + 1); i++) {
        for (int j = cy_y * tidy; j < cy_y * (tidy + 1); j++) {
            if ((tidx == 0 && i == 0) || (tidy == 0 && j == 0))
                continue;
            m_arrays.E[i * m_arrays.e_pitch + j * m_arrays.E_size_z] = (m_arrays.Ex[(i-1) * m_arrays.ex_pitch + j] + m_arrays.Ex[i * m_arrays.ex_pitch + j]) / 2.0;
            m_arrays.E[i * m_arrays.e_pitch + j * m_arrays.E_size_z + 1] = (m_arrays.Ey[i * m_arrays.ey_pitch + j - 1] + m_arrays.Ey[i * m_arrays.ey_pitch + j]) / 2.0;

            E_mag[tidy * gridDim.x * blockDim.x + tidx] += sqrt((m_arrays.E[i * m_arrays.e_pitch + j * m_arrays.E_size_z] * m_arrays.E[i * m_arrays.e_pitch + j * m_arrays.E_size_z])
                                                            + (m_arrays.E[i * m_arrays.e_pitch + j * m_arrays.E_size_z + 1] * m_arrays.E[i *m_arrays. e_pitch + j * m_arrays.E_size_z + 1]));
        }
    }

    for (int i = cx_x * tidx; i < cx_x * (tidx + 1); i++) {
        for (int j = cy_y * tidy; j < cy_y * (tidy + 1); j++) {
            if ((tidx == 0 && i == 0) || (tidy == 0 && j == 0))
                continue;
            m_arrays.B[i * m_arrays.b_pitch + j * m_arrays.B_size_z + 2] = (m_arrays.Bz[(i-1) * m_arrays.bz_pitch + j] + m_arrays.Bz[i * m_arrays.bz_pitch + j] + m_arrays.Bz[i * m_arrays.bz_pitch + j - 1]
                                                                            + m_arrays.Bz[(i-1) * m_arrays.bz_pitch + j - 1]) / 4.0;

            B_mag[tidy * gridDim.x * blockDim.x + tidx] += sqrt(m_arrays.B[i * m_arrays.b_pitch + j * m_arrays.B_size_z + 2] * m_arrays.B[i * m_arrays.b_pitch + j * m_arrays.B_size_z + 2]);
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
	problem_set_up<<<1,1>>>(m_arrays, m_variables);
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
		apply_boundary<<<grid, block>>>(m_arrays);
		update_fields<<<grid, block>>>(m_constants, m_arrays, m_variables);

		t += dt;

		if (i % output_freq == 0) {
			double E_mag = 0, B_mag = 0;
			resolve_to_grid<<<grid, block>>>(d_E_mag_cudaV, d_B_mag_cudaV, m_arrays);
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
	resolve_to_grid<<<grid, block>>>(&E_mag, &B_mag, m_arrays);
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


