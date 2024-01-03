#include <mpi.h>
#include <stdint.h>
#include "include/utils.h"
#include "include/curlutils.h"

//remind me to never write code using mpi again.

int main(int argc, char* argv[]){
    bool isFile = true;
    ARRAY myData;
    int n_procs, my_rank, root = 0;
    size_t elements, k_select;
    uint32_t value, *data = NULL;
    double max_time, min_time, avg_time, local_time, start, stop;
    int *send_counts = NULL, *displacements = NULL;
    char *inputFilePath, *URL;
    //MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Hello from processor %s, rank %d out of %d processors\n",processor_name, my_rank, n_procs);
    //initial data loading and argument parsing
    if (argc < 4) {
        printf("Invalid argument count! Exiting!\n");
        printCommandUsage();
        return 1;
    }
    else{
        if(argv[1][0] == '-'){
            switch(argv[1][1]){
                    case 'f': //file mode
                    isFile = true;
                    inputFilePath = argv[2];
                    elements = atoi(argv[3]);
                    k_select = atoi(argv[4]) - 1;
                    break;
                case 'l': //URL mode
                    isFile = false;
                    URL = argv[2];
                    k_select = atoi(argv[3]) - 1;
                    break;
                default: //invalid parameter
                    printf("Invalid parameter! Exiting!\n");
                    printCommandUsage();
                    return 1;
            }
        }
        else{
            if(argc == 4){
                isFile = true;
                inputFilePath = argv[1];
                elements = atoi(argv[2]);
                k_select = atoi(argv[3]) - 1;
            }
            else{
                 printf("Invalid parameter! Exiting!\n");
                 printCommandUsage();
                 return 1;
            }
        }
        if(k_select < 0){
            printf("Error! invalid k-th value. K-th value muse be positive and smaller or equal to the amount of elements.\n");
            return 2;
        }
    }
    if(isFile){
        FILE* file = fopen(inputFilePath,"r");
        if(file == NULL){
            printf("Error! File not found!\n");
            return 3;
        }
        data = malloc(elements*sizeof(uint32_t));
        // I could read in parallel (possible #TODO)
        double read_temp;
        for(int i = 0; i < elements;++i){
            fscanf(file, "%lg", &read_temp);
            data[i] = (uint32_t) read_temp;
        }
        fclose(file);
        size_t leftovers,elementsPerProcess,chunkSize;
        //Distribute the data, leftovers as well
        elementsPerProcess = elements / n_procs;
        leftovers = elements % n_procs;
        chunkSize = (my_rank < leftovers)? elementsPerProcess + 1 : elementsPerProcess;
        send_counts = malloc(n_procs*sizeof(int));
        displacements = malloc(n_procs*sizeof(int));
        int currDisplacement = 0;
        for(int i = 0; i < n_procs;++i){
            send_counts[i] = (i < leftovers)? elementsPerProcess +1 : elementsPerProcess;
            displacements[i] = currDisplacement;
            currDisplacement += send_counts[i];
        }
        uint32_t* local_data = malloc((chunkSize)*sizeof(uint32_t));
        //prepare for deployment, synchronize at start
        //scatter data to distributed memory, even when they're not split even
        MPI_Scatterv(data,send_counts, displacements,MPI_UINT32_T, local_data, chunkSize,MPI_UINT32_T, root, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if(my_rank == root){
            printf("-------------------------QUICKSELECT MPI START-------------------------\n");
        }
        start = MPI_Wtime();
        value = quickselectMPI2(root, local_data, chunkSize,0, chunkSize-1, k_select);
    }
    else{
        //prepare for deployment, synchronize at start
        //distribute bytes of remote file among processors
        myData = getWikiPartition(URL, my_rank, n_procs);
        MPI_Barrier(MPI_COMM_WORLD);
        if(my_rank == root){
            printf("-------------------------QUICKSELECT MPI START-------------------------\n");
        }
        start = MPI_Wtime();
        value = quickselectMPI2(root, myData.data, myData.size, 0, myData.size-1, k_select);
    }
    printf("rank %d selected value is %u\n",my_rank,value);
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
