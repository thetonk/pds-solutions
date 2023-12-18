#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "include/utils.h"

//remind me to never write code using mpi again, you better kill me if I do.

int main(int argc, char* argv[]){
    int n_procs, my_rank, root = 0, elements = 6,k_select=5, value, chunkSize;
    double max_time, min_time, avg_time, local_time, start, stop;
    int* data = NULL;
    char *inputFilePath;
    //initial data loading and argument parsing
    if (argc < 4) {
        printf("Invalid argument count! Exiting!\n");
        printf("[USAGE] ./test-openmpi <input file path> <number of elements> <element position to search>\n");
        return 1;
    }
    else{
        inputFilePath = argv[1];
        elements = atoi(argv[2]);
        k_select = atoi(argv[3]) - 1;
    }
    FILE* file = fopen(inputFilePath,"r");
    if(file == NULL){
        printf("Error! File not found!\n");
        return 2;
    }
    data = malloc(elements*sizeof(int));
    // I could read in parallel (possible #TODO)
    double read_temp;
    for(int i = 0; i < elements;++i){
        fscanf(file, "%lg", &read_temp);
        data[i] = (int) read_temp;
    }
    fclose(file);
    //MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    chunkSize = elements/n_procs;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Hello from processor %s, rank %d out of %d processors\n",processor_name, my_rank, n_procs);
    //prepare for deployment, synchronize at start
    int* local_data = malloc((chunkSize)*sizeof(int));
    //scatter data to distributed memory
    MPI_Scatter(data, chunkSize, MPI_INT, local_data, chunkSize, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank == root){
        printf("-------------------------QUICKSELECT MPI START-------------------------\n");
    }
    start = MPI_Wtime();
    value = quickselectMPI2(root, local_data, chunkSize,0, chunkSize-1, k_select);
    printf("rank %d selected value is %d\n",my_rank,value);
    printf("process with rank %d finished!\n", my_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank == root){
        printf("-------------------------QUICKSELECT MPI STOP-------------------------\n");
    }
    stop = MPI_Wtime();
    local_time = stop - start;
    //statistics
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if(my_rank == root){
        avg_time = avg_time / n_procs;
        printf("MPI Jobs times (in seconds):\nAverage time: %f, max time: %f, min time: %f\n",avg_time, max_time,min_time);
    }
    MPI_Finalize();
    return 0;
}
