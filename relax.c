#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

double **testingArray;

typedef struct
{
    int secs;
    int usecs;
} TIMEDIFF;

// Timer function that returns time in seconds and microseconds between the 2 time given values
TIMEDIFF *diffTime(struct timeval *start, struct timeval *end)
{
    TIMEDIFF *diff = (TIMEDIFF *)malloc(sizeof(TIMEDIFF));

    if (start->tv_sec == end->tv_sec)
    {
        diff->secs = 0;
        diff->usecs = end->tv_usec - start->tv_usec;
    }
    else
    {
        diff->usecs = 1000000 - start->tv_usec;
        diff->secs = end->tv_sec - (start->tv_sec + 1);
        diff->usecs += end->tv_usec;
        if (diff->usecs >= 1000000)
        {
            diff->usecs -= 1000000;
            diff->secs += 1;
        }
    }

    return diff;
}

// Sequential version of algorithm for purpose of comparison
double runSeq(int as, double tol, int verbose)
{
    double **masterArraySeq;
    double **dummyArraySeq;
    double maxDiff = 9999;

    masterArraySeq = (double **)malloc(as * sizeof(double *));
    dummyArraySeq = (double **)malloc(as * sizeof(double *));

    for (int i = 0; i < as; i++)
    {
        masterArraySeq[i] = (double *)malloc(as * sizeof(double));
        dummyArraySeq[i] = (double *)malloc(as * sizeof(double));
    }

    for (int i = 0; i < as; i++)
    {
        for (int j = 0; j < as; j++)
        {
            masterArraySeq[i][j] = testingArray[i][j];
        }
    }

    // Begin timer
    struct timeval myTVstart, myTVend;
    TIMEDIFF *difference;
    gettimeofday(&myTVstart, NULL);

    while (maxDiff > tol)
    {
        maxDiff = 0;

        // Averaging 4 adjacent numbers for each and storing the result in a temporary array
        for (int i = 1; i < as - 1; i++)
        {
            for (int j = 1; j < as - 1; j++)
            {
                dummyArraySeq[i][j] = (masterArraySeq[i - 1][j] +
                                       masterArraySeq[i + 1][j] +
                                       masterArraySeq[i][j - 1] +
                                       masterArraySeq[i][j + 1]) /
                                      4;
            }
        }

        // Updating main array once all averages have been calculated
        // Also checking differences between all pairs of elements to see if within tolerance
        for (int i = 1; i < as - 1; i++)
        {
            for (int j = 1; j < as - 1; j++)
            {
                if (fabs(masterArraySeq[i][j] - dummyArraySeq[i][j]) > maxDiff)
                {
                    maxDiff = fabs(masterArraySeq[i][j] - dummyArraySeq[i][j]);
                }
                masterArraySeq[i][j] = dummyArraySeq[i][j];
            }
        }
    }

    // End timer
    gettimeofday(&myTVend, NULL);
    difference = diffTime(&myTVstart, &myTVend);

    if (verbose == 1)
    {
        printf("\nResult:\n");
        // Output result to console
        for (int i = 0; i < as; i++)
        {
            for (int j = 0; j < as; j++)
            {
                printf("%0.3f ", masterArraySeq[i][j]);
            }
            printf("\n");
        }
    }

    // Free memory used for arrays
    for (int i = 0; i < as; i++)
    {
        free(masterArraySeq[i]);
        free(dummyArraySeq[i]);
    }
    free(masterArraySeq);
    free(dummyArraySeq);

    return difference->secs + ((double)difference->usecs / 1000000);
}

double run(int worldRank, int worldSize, int as, double tol, int verbose)
{
    MPI_Status stat;

    double **masterArray;
    double **buffer;
    double **updatedBuffer;

    double diff = 0;
    int loop = 1;

    //(currently need exact divisors)
    int rowsPerProc = ((as - 2) / worldSize) + 2;

    // Initialise buffers
    buffer = (double **)malloc(rowsPerProc * sizeof(double *));
    updatedBuffer = (double **)malloc((rowsPerProc - 2) * sizeof(double *));

    for (int i = 0; i < rowsPerProc; i++)
    {
        buffer[i] = (double *)malloc(as * sizeof(double));
        if (i != 0 && i != rowsPerProc - 1)
            updatedBuffer[i - 1] = (double *)malloc(as * sizeof(double));
    }

    if (worldRank == 0)
    {
        // Root process initialises master array
        masterArray = (double **)malloc(as * sizeof(double *));
        for (int i = 0; i < as; i++)
        {
            masterArray[i] = (double *)malloc(as * sizeof(double));
            for (int j = 0; j < as; j++)
                masterArray[i][j] = testingArray[i][j];
        }
    }

    while (loop == 1)
    {
        loop = 0;

        if (worldRank == 0)
        {
            // Send lines of array to slave processors...
            for (int i = 1; i < worldSize; i++)
                for (int j = 0; j < rowsPerProc; j++)
                {
                    int rowToSend = ((as - 2) / worldSize) * i + j;
                    MPI_Send(masterArray[rowToSend], as, MPI_DOUBLE, i, j, MPI_COMM_WORLD);
                }

            // ...and own buffer
            for (int i = 0; i < rowsPerProc; i++)
                for (int j = 0; j < as; j++)
                    buffer[i][j] = masterArray[i][j];
        }

        // All non-root processors recieve their rows
        else
            for (int i = 0; i < rowsPerProc; i++)
                MPI_Recv(buffer[i], as, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &stat);

        for (int i = 1; i < rowsPerProc - 1; i++)
        {
            diff = 0;

            // Compute and store averages for all non-edges
            for (int j = 1; j < as - 1; j++)
            {
                updatedBuffer[i - 1][j] = (buffer[i + 1][j] + buffer[i - 1][j] + buffer[i][j + 1] + buffer[i][j - 1]) / 4;
                if (fabs(updatedBuffer[i - 1][j] - buffer[i][j]) > diff)
                    diff = fabs(updatedBuffer[i - 1][j] - buffer[i][j]);
            }

            // Store edge values as given
            updatedBuffer[i - 1][0] = buffer[i][0];
            updatedBuffer[i - 1][as - 1] = buffer[i][as - 1];

            // If maximum difference between original and updated values is above tolerance, make first entry negative
            if (diff > tol)
                updatedBuffer[i - 1][0] = -1 * updatedBuffer[i - 1][0];

            // Send updated values to master array
            if (worldRank == 0)
            {
                for (int j = 0; j < as - 1; j++)
                    masterArray[i][j] = updatedBuffer[i - 1][j];

                for (int j = 1; j < worldSize; j++)
                    MPI_Recv(masterArray[j * (rowsPerProc - 2) + i], as, MPI_DOUBLE, j, j * (rowsPerProc - 2) + i, MPI_COMM_WORLD, &stat);
            }
            else
                MPI_Send(updatedBuffer[i - 1], as, MPI_DOUBLE, 0, worldRank * (rowsPerProc - 2) + i, MPI_COMM_WORLD);
        }

        // If any values are negative, flip them and re run loop
        if (worldRank == 0)
        {
            for (int i = 1; i < as - 1; i++)
                if (masterArray[i][0] < 0)
                {
                    masterArray[i][0] = -1 * masterArray[i][0];
                    loop = 1;
                }
        }
        MPI_Bcast(&loop, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Output result to console
    if (verbose == 1 && worldRank == 0)
        for (int i = 0; i < as; i++)
        {
            for (int j = 0; j < as; j++)
                printf("%0.3f ", masterArray[i][j]);
            printf("\n");
        }

    // Free memory used for arrays

    for (int i = 0; i < rowsPerProc; i++)
    {
        free(buffer[i]);
        if (i != 0 && i != rowsPerProc - 1)
            free(updatedBuffer[i - 1]);
    }
    free(buffer);
    free(updatedBuffer);

    if (worldRank == 0)
    {
        for (int i = 0; i < as; i++)
            free(masterArray[i]);
        free(masterArray);
    }

    return 0;
}

int main()
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    srand(time(0));
    rand();
    double t;

    //// Testing parameters ////
    double arraySize = 10;
    double tolerance = 0.01;
    int numTests = 1;
    int verbose = 1;
    ////////////////////////////

    // Create and initialise testing array that all tests will use
    if (world_rank == 0)
    {
        testingArray = (double **)malloc(arraySize * sizeof(double *));
        if (verbose == 1)
            printf("Initial array:\n");

        for (int i = 0; i < arraySize; i++)
        {
            testingArray[i] = (double *)malloc(arraySize * sizeof(double));
            for (int j = 0; j < arraySize; j++)
            {
                testingArray[i][j] = (double)rand() / (double)RAND_MAX;
                if (verbose == 1)
                    printf("%0.3f ", testingArray[i][j]);
            }
            if (verbose == 1)
                printf("\n");
        }
        if (verbose == 1)
            printf("\n");

        // Run sequential algorithm on testing array
        t = 0;
        for (int i = 0; i < numTests; i++)
            t += runSeq(arraySize, tolerance, verbose);
        printf("seq %f\n", t / numTests);
    }

    // Run distributed algorithm on testing array
    MPI_Barrier(MPI_COMM_WORLD);
    t = 0;
    for (int i = 0; i < numTests; i++)
        t += run(world_rank, world_size, arraySize, tolerance, verbose);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("MPI %f from %d\n", t / numTests, world_rank);

    MPI_Finalize();
}