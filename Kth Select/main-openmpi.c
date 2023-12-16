#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdbool.h>
#include "include/utils.h"

//remind me to never write code using mpi again, you better kill me if I do.

int main(int argc, char* argv[]){
    int n_procs, my_rank, root = 0, elements = 6,k_select=2, value;
    int* data = NULL;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Hello from processor %s, rank %d out of %d processors\n",processor_name, my_rank, n_procs);
    //prepare for deployment, synchronize at start
    if(my_rank == root){
        FILE* file = fopen("data/input3.txt", "r");
        if(file == NULL){
            printf("Error! File not found!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        data = malloc(elements*sizeof(int));
        // I could read in parallel (possible #TODO)
        double read_temp;
        for(int i = 0; i < elements;++i){
            fscanf(file, "%lg", &read_temp);
            data[i] = (int) read_temp;
        }
        fclose(file);
    }
    int* local_data = malloc((elements/n_procs)*sizeof(int));
    //scatter data to distributed memory
    MPI_Scatter(data, elements/n_procs, MPI_INT, local_data, elements/n_procs, MPI_INT, root, MPI_COMM_WORLD);
    value = quickselectMPI2(root, local_data, 0, (elements/n_procs)-1, k_select);
    printf("rank %d selected value is %d\n",my_rank,value);
    printf("process with rank %d finished!\n", my_rank);
    /*if(my_rank == root){
        //MPI_Recv(&value, 1, MPI_INT, root, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("The %d th value is %d\n", k_select,value);
        free(data);
    }*/
    MPI_Finalize();
    return 0;
}
