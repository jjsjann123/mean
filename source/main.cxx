/*!
 *
 *	CS 595 Assignment 9
 *	Jie Jiang
 *	
 */
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <fstream>

#include <vector>

using namespace std;

struct Vec3f{
	float x;
	float y;
	float z;
};

struct Vec3i{
	int x;
	int y;
	int z;
};

struct ChunkData{
	Vec3i offset;
	Vec3i range;
};

int main(int argc, char** argv)
{
	int myrank;
	int npes;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int xPartition, yPartition, zPartition;
	char *filename;
	
	if (argc != 5)
	{
		if ( myrank == 0 )
		{
			cout << "arguments not match, error..." << endl;
			cout << "usage: <program> <data file>  <xpartitions> <ypartitions> <zpartitions>" << endl;
		}
		return 1;
	}
	else
	{
		filename = argv[1];
		xPartition = atoi(argv[2]);
		yPartition = atoi(argv[3]);
		zPartition = atoi(argv[4]);
	}

	if (xPartition*yPartition*zPartition != npes-1)
	{
		if ( myrank == 0 )
		{
			cout << "Number of subvolume is not the same as the number of processes" << endl;
		}
		return 1;
	}
	int dim[3];


	float *buffer;
	double *meanBuffer;
	int *chunkListBuffer;
	int chunkBuffer[6];
	if ( myrank == 0)
	{
		// Load the data and get dimension out
		ifstream input (filename, ios::binary);
		input.read( (char*) &dim[0], sizeof(int));
		input.read( (char*) &dim[1], sizeof(int));
		input.read( (char*) &dim[2], sizeof(int));
		MPI_Bcast(dim, 3, MPI_INT, myrank, MPI_COMM_WORLD);
		meanBuffer = (double *) malloc ( npes * sizeof(double));
		Vec3i chunkSize;
		Vec3i chunkOffset;

		chunkListBuffer = (int *) malloc ( npes * 6 * sizeof ( int ) );

		// Calculate chunk size and range
		for ( int x = 0; x < xPartition; x++)
		{
			if ( x == xPartition -1 )
				chunkSize.x = dim[0] % xPartition + dim[0] / xPartition;
			else
				chunkSize.x = dim[0] / xPartition;
			chunkOffset.x = x * ( dim[0] / xPartition );
			for ( int y = 0; y < yPartition; y++)
			{
				if ( y == yPartition -1 )
					chunkSize.y = dim[1] % yPartition + dim[1] / yPartition;
				else
					chunkSize.y = dim[1] / yPartition;
				chunkOffset.y = y * ( dim[1] / yPartition );
				for ( int z = 0; z < zPartition; z++)
				{
					if ( z == zPartition -1 )
						chunkSize.z = dim[2] % zPartition + dim[2] / zPartition;
					else
						chunkSize.z = dim[2] / zPartition;
					chunkOffset.z = z * ( dim[2] / zPartition );
					int index = x+xPartition*(y+z*yPartition);
					int offset = (index + 1) * 6;
					chunkListBuffer[offset] = chunkOffset.x;
					chunkListBuffer[offset+1] = chunkOffset.y;
					chunkListBuffer[offset+2] = chunkOffset.z;
					chunkListBuffer[offset+3] = chunkSize.x;
					chunkListBuffer[offset+4] = chunkSize.y;
					chunkListBuffer[offset+5] = chunkSize.z;
					printf ("Subvolume <%d, %d>, <%d, %d>, <%d, %d> is assigned to process <%d>\n",
							chunkOffset.x, chunkOffset.x + chunkSize.x - 1,
							chunkOffset.y, chunkOffset.y + chunkSize.y - 1,
							chunkOffset.z, chunkOffset.z + chunkSize.z - 1,
							index);
				}
			}
		}

		MPI_Scatter(chunkListBuffer, 6, MPI_FLOAT, chunkBuffer, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);

		int bufferSize = dim[0]*dim[1];
		buffer = (float*) malloc (bufferSize*sizeof(float));
		for ( int i = 0; i <= dim[2]; i++)
		{
			input.read( (char*) buffer, dim[0]*dim[1]*sizeof(float));
			MPI_Bcast(buffer, bufferSize, MPI_INT, myrank, MPI_COMM_WORLD);
		}

		double mean = 0;

		MPI_Gather(&mean, 1, MPI_DOUBLE, meanBuffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		printf ( "process 0 receives local means ");
		for ( int i = 1; i < npes; i++)
		{
			printf (" <%f> ", buffer[i]);
			mean += meanBuffer[i] * chunkListBuffer[i*6+3] * chunkListBuffer[i*6+4] * chunkListBuffer[i*6+5];
		}
		mean /= dim[0] * dim[1] * dim[2];
		printf (" and the overall mean = <%f>", mean);
		free(buffer);
		free(chunkListBuffer);
	} else { // myrank != 0
		// Receive data dimension from root
		MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatter(chunkListBuffer, 6, MPI_FLOAT, chunkBuffer, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);

		//printf( "Hello world from process %d of %d: dim %d, %d, %d; chunk: %d %d %d %d %d %d\n", myrank, npes, dim[0], dim[1], dim[2], chunkBuffer[0], chunkBuffer[1], chunkBuffer[2], chunkBuffer[3], chunkBuffer[4], chunkBuffer[5]);
		int dataSize = chunkBuffer[3] * chunkBuffer[4] * chunkBuffer[5];
		int *localData = (int *) malloc ( dataSize * sizeof(int));

		int bufferSize = dim[0]*dim[1];
		buffer = (float*) malloc (bufferSize*sizeof(float));
		int index = 0;
		for ( int i = 0; i <= dim[2]; i++)
		{
			MPI_Bcast(buffer, bufferSize, MPI_INT, 0, MPI_COMM_WORLD);
			if ( i >= chunkBuffer[2] && i < chunkBuffer[2] + chunkBuffer[5])
			{
				for ( int v = chunkBuffer[1]; v < chunkBuffer[1] + chunkBuffer[4]; v++)
				{
					for ( int u = chunkBuffer[0]; u < chunkBuffer[0] + chunkBuffer[3]; u++)
					{
						int position = u + dim[0]*v;
						localData[index] = buffer[position];
						index++;
					}
				}
			}
		}
		if (index != dataSize)
		{
			// Error data split messed up.
			printf( "process: %d : index: %d dataSize: %d", myrank, index, dataSize);
		}
		double mean = 0;
		for ( int i = 0; i < dataSize; i++)
		{
			mean += localData[i];
		}
		printf ( "process %d data size: %d total %f \n", myrank, dataSize, mean);
		mean /= dataSize;
		printf ("Process <%d> has data <%d, %d>, <%d, %d>, <%d, %d>, mean = <%f> \n",
									myrank,
									chunkBuffer[0], chunkBuffer[0] + chunkBuffer[3]-1,
									chunkBuffer[1], chunkBuffer[1] + chunkBuffer[4]-1,
									chunkBuffer[2], chunkBuffer[2] + chunkBuffer[5]-1,
									mean);
		MPI_Gather(&mean, 1, MPI_DOUBLE, meanBuffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		free(localData);
		free(buffer);
	}
	MPI_Finalize();
	return 0;
}
