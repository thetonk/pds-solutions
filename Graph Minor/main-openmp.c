#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include "mmio.h"
#include <string.h>

#define MAX_THREADS 5

struct ThreadData{
    int threadID;
    int non_zeros;
    int *non_zeros_indexes;
};

struct row{
    int non_zeros;
    int *columns, *values;
};

void printrows(struct row *matrix, int rowcount){
    for(int i = 0; i< rowcount; ++i){
        for (int j = 0; j < matrix[i].non_zeros; ++j){
            printf("(%d,%d) %d\n", i, matrix[i].columns[j], matrix[i].values[j]);
        }
    }
}

//trying to save as much memory as possible, adds time complexity tho
void saveResult(struct row* r, int col, int v){
    if(r->non_zeros == 0){
        r->columns = malloc(sizeof(int));
        r->values = malloc(sizeof(int));
        r->columns[0] = col;
        r->values[0] = v;
        r->non_zeros++;
    }
    else{
        r->non_zeros++;
        r->columns = realloc(r->columns, r->non_zeros*sizeof(int));
        r->values = realloc(r->values, r->non_zeros*sizeof(int));
        r->columns[r->non_zeros-1] = col;
        r->values[r->non_zeros-1] = v;
    }
}

int main(int argc, char *argv[]){
    int CLUSTERS, *c, *rowsA ,*colsA, *valA, NON_ZEROS;
    int COL,ROW_COUNT, NON_ZEROS_PER_THREAD[MAX_THREADS];
    double read_temp;
    char *matrixFilePath, *clusterVectorPath;
    if(argc == 3){
        //use user defined paths from command line
        matrixFilePath = argv[1];
        clusterVectorPath = argv[2];
    } 
    else{
        //use our default
        matrixFilePath = "data/dwt_2680.mtx";
        clusterVectorPath = "data/out10-dwt.txt";
    }
    MM_typecode matcode;
    FILE* f = fopen(matrixFilePath,"r");
    FILE* c_f = fopen(clusterVectorPath,"r");
    if(f == NULL || c_f == NULL){
        printf("Error! input files not found!\n");
        exit(1);
    }
    mm_read_banner(f,&matcode);
    mm_read_mtx_crd_size(f,&ROW_COUNT,&COL,&NON_ZEROS);
    c = malloc(ROW_COUNT*sizeof(int));
    rowsA = malloc(NON_ZEROS*sizeof(int));
    colsA = malloc(NON_ZEROS*sizeof(int));
    valA = malloc(NON_ZEROS*sizeof(int));
    memset(c,0, ROW_COUNT*sizeof(int)); //just in case
    memset(NON_ZEROS_PER_THREAD,0,MAX_THREADS*sizeof(int));
    //reading input files
    for(int i = 0; i < ROW_COUNT;++i){
        fscanf(c_f,"%lg",&read_temp);
        c[i] = (int) read_temp;
        if(c[i] > CLUSTERS){
            CLUSTERS = c[i];
        }
    }
    fclose(c_f);
    for(int i = 0; i < NON_ZEROS; ++i){
        if(mm_is_pattern(matcode)){
            fscanf(f,"%d %d\n", &rowsA[i],&colsA[i]);
            valA[i] = 1;
        }
        else{
            fscanf(f,"%d %d %lg\n", &rowsA[i],&colsA[i], &read_temp);
            if (read_temp < 0) read_temp = -read_temp;
            if((int) read_temp == 0) read_temp = 1; 
            valA[i] = read_temp;
        }
        NON_ZEROS_PER_THREAD[(c[rowsA[i] -1] -1) % MAX_THREADS]++;
    }
    fclose(f);
    //end reading input files
    //data initialization
    struct row *minor;
    minor = malloc(CLUSTERS*sizeof(struct row));
    for(int i = 0; i<CLUSTERS;++i){
        minor[i].non_zeros = 0;
    }
    int counter = 0;
    struct timeval start,end;
    struct ThreadData threadData[MAX_THREADS];
    //overhead
    for(int i = 0; i < MAX_THREADS; ++i){
        threadData[i].threadID = i;
        threadData[i].non_zeros_indexes = malloc(NON_ZEROS_PER_THREAD[i]*sizeof(int));
        threadData[i].non_zeros = NON_ZEROS_PER_THREAD[i];
        counter = 0;
        for(int j = 0; j < NON_ZEROS; ++j){
            //split data among threads according to which row they belong in cluster adjacency matrix
            if((c[rowsA[j] -1 ] -1) % MAX_THREADS == i){
                threadData[i].non_zeros_indexes[counter] = j;
                counter++;
            }
        }
    }
    //end overhead
    //start threads
    omp_set_dynamic(0);
    omp_set_num_threads(MAX_THREADS);
    gettimeofday(&start,0);
    #pragma omp parallel
    {
        //thread code region
        int dest_row, dest_col, minor_non_zeros,index;
        int threadID = omp_get_thread_num();
        bool stored = false;
        //each thread has its own data in the threadData array, index is selected according to its ID.
        for(int i = 0 ; i < threadData[threadID].non_zeros; ++i){
            stored = false;
            index = threadData[threadID].non_zeros_indexes[i];
            dest_row = c[rowsA[index]-1] -1;
            dest_col = c[colsA[index]-1]-1;
            minor_non_zeros = minor[dest_row].non_zeros;
            if(minor_non_zeros == 0){
                saveResult(&minor[dest_row], dest_col, valA[index]);
                stored = true;
                
            }
            else{
                for(int j = 0; j < minor_non_zeros;++j){
                    if(minor[dest_row].columns[j] == dest_col){
                        minor[dest_row].values[j] += valA[index];
                        stored = true;
                    }
                }
            }
            if(!stored){ //new column found
                saveResult(&minor[dest_row], dest_col,valA[index]);
            }
        }
        //this array is not needed anymore
        free(threadData[threadID].non_zeros_indexes);
    }
    gettimeofday(&end, 0);
    //memory cleanup
    free(rowsA);
    free(colsA);
    free(valA);
    //show results
    printf("-------------\nminor matrix:\n");
    printrows(minor,CLUSTERS);
    long secondsElapsed = end.tv_sec - start.tv_sec;
    long microsecondsElapsed = end.tv_usec - start.tv_usec;
    double totalTime = secondsElapsed + 1e-6*microsecondsElapsed;
    printf("Processing time: %f seconds\n",totalTime);
    for(int i = 0; i < CLUSTERS;++i){
        free(minor[i].columns);
        free(minor[i].values);
    }
    free(minor);
    return 0;
}