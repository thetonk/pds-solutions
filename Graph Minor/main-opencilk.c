#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "mmio.h"
#include <cilk/cilk.h>
#include <sys/time.h>


struct ThreadData{
    int threadID;
    int non_zeros;
    int *non_zeros_indexes;
};

struct row{
    int non_zeros;
    int *columns, *values;
};

struct row *minor;
int CLUSTERS, *c, *rowsA ,*colsA, *valA, NON_ZEROS;

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

void graphMinorRe(struct ThreadData *myData){
    int dest_row, dest_col, minor_non_zeros,index;
    bool stored = false;
    for(int i = 0 ; i < myData->non_zeros; ++i){
        stored = false;
        index = myData->non_zeros_indexes[i];
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
    free(myData->non_zeros_indexes);
}

int main(int argc, char *argv[]) {
    struct timeval start,end;
    int COL,ROW_COUNT, MAX_THREADS,file_non_zeros;
    double read_temp;
    char *matrixFilePath, *clusterVectorPath;
    switch(argc){
        //use user defined paths from command line
        case 4:
            MAX_THREADS = atoi(argv[3]);
            matrixFilePath = argv[1];
            clusterVectorPath = argv[2];
            break;
        case 3:
            matrixFilePath = argv[1];
            clusterVectorPath = argv[2];
            break;
        //or use our defaults
        default:
            MAX_THREADS = 7;
            matrixFilePath = "data/blckhole.mtx";
            clusterVectorPath = "data/out7.txt";
    }
    MM_typecode matcode;
    FILE* f = fopen(matrixFilePath,"r");
    FILE* c_f = fopen(clusterVectorPath,"r");
    if(f == NULL || c_f == NULL){
        printf("Error! input files not found!\n");
        exit(1);
    }
    mm_read_banner(f,&matcode);
    mm_read_mtx_crd_size(f,&ROW_COUNT,&COL,&file_non_zeros);
    if(mm_is_hermitian(matcode)){
        printf("Error! Hermitian matrices are not supported!\n");
        exit(2);
    }
    if(mm_is_symmetric(matcode) || mm_is_skew(matcode)){
        NON_ZEROS = 2*file_non_zeros; //make it double temporarily, because file contains only elements of below triangle of matrix
    }
    else{
        NON_ZEROS = file_non_zeros;
    }
    c = malloc(ROW_COUNT*sizeof(int));
    rowsA = malloc(NON_ZEROS*sizeof(int));
    colsA = malloc(NON_ZEROS*sizeof(int));
    valA = malloc(NON_ZEROS*sizeof(int));
    memset(c,0, ROW_COUNT*sizeof(int)); //just in case
    int NON_ZEROS_PER_THREAD[MAX_THREADS];
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
    int counter = 0, actual_non_zeros = 0;
    for(int i = 0; i < file_non_zeros; ++i){
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
        if(mm_is_symmetric(matcode)){
            if(rowsA[i] != colsA[i]){
                rowsA[file_non_zeros+counter] = colsA[i];
                colsA[file_non_zeros+counter] = rowsA[i];
                valA[file_non_zeros+counter] = valA[i];
                NON_ZEROS_PER_THREAD[(c[rowsA[file_non_zeros+counter] -1] -1) % MAX_THREADS]++;
                counter++;
                actual_non_zeros += 2;
            }
            else{ //diagonal element
                actual_non_zeros++;
            }
        }
        else if(mm_is_skew(matcode)){
            if(rowsA[i] != colsA[i]){
                rowsA[file_non_zeros+counter] = colsA[i];
                colsA[file_non_zeros+counter] = rowsA[i];
                valA[file_non_zeros+counter] = -valA[i];
                NON_ZEROS_PER_THREAD[(c[rowsA[file_non_zeros+counter] -1] -1) % MAX_THREADS]++;
                counter++;
                actual_non_zeros += 2;
            }
            else{ //diagonal element
                actual_non_zeros++;
            }
        }
        NON_ZEROS_PER_THREAD[(c[rowsA[i] -1] -1) % MAX_THREADS]++;
    }
    fclose(f);
    if(mm_is_symmetric(matcode) || mm_is_skew(matcode)){
        NON_ZEROS = actual_non_zeros;
        //save memory
        rowsA = realloc(rowsA, NON_ZEROS*sizeof(int));
        colsA = realloc(colsA, NON_ZEROS*sizeof(int));
        valA = realloc(valA,NON_ZEROS*sizeof(int));
    }
    //end reading input files
    //data initialization
    minor = malloc(CLUSTERS*sizeof(struct row));
    for(int i = 0; i<CLUSTERS;++i){
        minor[i].non_zeros = 0;
    }
    counter = 0;
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
    //printf("clusters %i, rows %i columns %i\n",CLUSTERS,ROW_COUNT,COL);
    //end overhead
    //start threads
    gettimeofday(&start, 0);
    cilk_scope{
        for(int i = 0; i < MAX_THREADS; ++i){
            cilk_spawn graphMinorRe(&threadData[i]);
        }
    }
    gettimeofday(&end,0);
    long secondsElapsed = end.tv_sec - start.tv_sec;
    long microsecondsElapsed = end.tv_usec - start.tv_usec;
    double totalTime = secondsElapsed + 1e-6*microsecondsElapsed;
    //show results
    printf("-------------\nminor matrix:\n");
    printrows(minor,CLUSTERS);
    //cleanup
    free(rowsA);
    free(colsA);
    free(valA);
    for(int i = 0; i < CLUSTERS;++i){
        free(minor[i].columns);
        free(minor[i].values);
    }
    free(minor);
    printf("Processing time: %f seconds\n",totalTime);
    return 0;
}
