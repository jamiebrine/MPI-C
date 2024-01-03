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
double runSeq(int as, double tol, int verbose, double **ta)
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
        for (int j = 0; j < as; j++)
            masterArraySeq[i][j] = ta[i][j];

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
        printf("Result Seq:\n");
        // Output result to console
        for (int i = 0; i < as; i++)
        {
            for (int j = 0; j < as; j++)
            {
                printf("%0.3f ", masterArraySeq[i][j]);
            }
            printf("\n");
        }
        printf("\n");
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

double run(int worldRank, int worldSize, int as, double tol, int verbose, double **ta)
{
    MPI_Status stat;

    double *masterArr;
    double **masterArray;

    double diff = 0;
    int loop = 1;

    struct timeval myTVstart, myTVend;
    TIMEDIFF *difference;

    // Assign each processor a number of rows it will work on
    int rowsPerProc[worldSize];
    int recvCounts[worldSize];
    int displacements[worldSize];

    for (int i = 0; i < worldSize; i++)
    {
        rowsPerProc[i] = ((as - 2) / worldSize) + 2;
        rowsPerProc[i] += (((as - 2) % worldSize) + i) / worldSize;
        recvCounts[i] = (rowsPerProc[i] - 2) * as;

        if (i == 0)
            displacements[i] = as;
        else
            displacements[i] = displacements[i - 1] + recvCounts[i - 1];
    }

    // Initialise buffers
    double *bufferArr = malloc(rowsPerProc[worldRank] * as * sizeof(double));
    double **buffer = malloc(rowsPerProc[worldRank] * sizeof(double *));

    double *uBufferArr = malloc((rowsPerProc[worldRank] - 2) * as * sizeof(double));
    double **updatedBuffer = malloc((rowsPerProc[worldRank] - 2) * sizeof(double *));

    for (int i = 0; i < rowsPerProc[worldRank]; i++)
    {
        buffer[i] = &(bufferArr[i * as]);
        if (i != 0 && i != rowsPerProc[worldRank] - 1)
            updatedBuffer[i - 1] = &(uBufferArr[(i - 1) * as]);
    }

    if (worldRank == 0)
    {
        // Root process initialises master array
        masterArr = malloc(as * as * sizeof(double));
        masterArray = malloc(as * sizeof(double *));

        for (int i = 0; i < as; i++)
        {
            masterArray[i] = &(masterArr[i * as]);

            for (int j = 0; j < as; j++)
                masterArray[i][j] = ta[i][j];
        }
        // Start timer
        gettimeofday(&myTVstart, NULL);
    }

    while (loop == 1)
    {
        loop = 0;

        if (worldRank == 0)
        {
            int first = rowsPerProc[worldRank] - 1;
            int extra;

            // Send lines of array to slave processors...
            for (int i = 1; i < worldSize; i++)
            {
                extra = (((as - 2) % worldSize) + i) / worldSize;
                for (int j = 0; j < rowsPerProc[worldRank] + extra; j++)
                {
                    int rowToSend = first - 1 + j;
                    MPI_Send(masterArray[rowToSend], as, MPI_DOUBLE, i, j, MPI_COMM_WORLD);
                }
                first += rowsPerProc[worldRank] - 2 + extra;
            }

            // ...and own buffer
            for (int i = 0; i < rowsPerProc[worldRank]; i++)
                for (int j = 0; j < as; j++)
                    buffer[i][j] = masterArray[i][j];
        }

        // All non-root processors recieve their rows
        else
            for (int i = 0; i < rowsPerProc[worldRank]; i++)
                MPI_Recv(buffer[i], as, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &stat);

        for (int i = 1; i < rowsPerProc[worldRank] - 1; i++)
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
        }

        // Recieve updated values from slave processes
        MPI_Gatherv(uBufferArr, (rowsPerProc[worldRank] - 2) * as,
                    MPI_DOUBLE, masterArr, recvCounts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

    // End timer
    if (worldRank == 0)
    {
        gettimeofday(&myTVend, NULL);
        difference = diffTime(&myTVstart, &myTVend);
    }

    // Output result to console
    if (verbose == 1 && worldRank == 0)
    {
        printf("Result MPI:\n");
        for (int i = 0; i < as; i++)
        {
            for (int j = 0; j < as; j++)
                printf("%0.3f ", masterArray[i][j]);
            printf("\n");
        }
        printf("\n");
    }

    // Free memory used for arrays
    free(bufferArr);
    free(buffer);
    free(uBufferArr);
    free(updatedBuffer);

    if (worldRank == 0)
    {
        free(masterArr);
        free(masterArray);
    }

    if (worldRank == 0)
        return difference->secs + ((double)difference->usecs / 1000000);
    else
        return 0;
}

int main()
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Start timer
    struct timeval overallTVstart, overallTVend;
    TIMEDIFF *overallTime;
    if (world_rank == 0)
        gettimeofday(&overallTVstart, NULL);

    ////////////////////////////////////////////
    //////////// Testing parameters ////////////
    const int numSizes = 2;
    const int arraySize[numSizes] = {10, 20};
    const double tolerance = 0.01;
    const int numTests = 3;
    const int verbose = 1;
    const int doSeq = 1;
    const int doMPI = 0;
    ////////////////////////////////////////////
    ////////////////////////////////////////////

    // Uses same random seed to ensure consistency between tests
    srand(0);
    rand();

    double **testingArray;
    double tMPI[numSizes] = {0};
    double tSeq[numSizes] = {0};

    for (int size = 0; size < numSizes; size++)
    {
        // Initialise testing array that all tests of that size will use
        if (world_rank == 0)
        {
            testingArray = (double **)malloc(arraySize[size] * sizeof(double *));
            for (int i = 0; i < arraySize[size]; i++)
                testingArray[i] = (double *)malloc(arraySize[size] * sizeof(double));
        }

        for (int i = 0; i < numTests; i++)
        {
            // Populate testing array with random values
            if (world_rank == 0)
            {
                if (verbose)
                    printf("Testing array:\n");

                for (int j = 0; j < arraySize[size]; j++)
                {
                    for (int k = 0; k < arraySize[size]; k++)
                    {
                        testingArray[j][k] = (double)rand() / (double)RAND_MAX;
                        if (verbose)
                            printf("%0.3f ", testingArray[j][k]);
                    }
                    if (verbose)
                        printf("\n");
                }

                if (verbose)
                    printf("\n");
            }

            // Run testing algorithms
            if (doSeq)
            {
                if (world_rank == 0)
                    tSeq[size] += runSeq(arraySize[size], tolerance, verbose, testingArray);
                MPI_Barrier(MPI_COMM_WORLD);
            }

            if (doMPI)
                tMPI[size] += run(world_rank, world_size, arraySize[size], tolerance, verbose, testingArray);
        }

        // Free memory used for testing array
        if (world_rank == 0)
        {
            for (int i = 0; i < arraySize[size]; i++)
                free(testingArray[i]);
            free(testingArray);
        }
    }

    // Stop timer
    if (world_rank == 0)
    {
        gettimeofday(&overallTVend, NULL);
        overallTime = diffTime(&overallTVstart, &overallTVend);
    }

    // Output results
    if (world_rank == 0)
    {
        for (int i = 0; i < numSizes; i++)
        {
            printf("%d:\n", arraySize[i]);
            if (doSeq)
                printf("Seq %f\n", tSeq[i] / numTests);
            if (doMPI)
                printf("MPI %f\n", tMPI[i] / numTests);
            printf("\n");
        }
        printf("(%f)\n", overallTime->secs + ((double)overallTime->usecs / 1000000));
    }

    MPI_Finalize();
}