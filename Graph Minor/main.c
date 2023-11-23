#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "mmio.h"
#include <string.h>
#include <sys/time.h>

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

//converts custom CSC format into COO, if needed, for the sake of completeness
void CSCtoCOO(struct row *matrix, int n, int* rows, int* columns,int* values){
    int non_zeros = 0, count = 0;
    // add the non zeros of each row
    for(int i = 0 ; i < n; ++i){
        non_zeros += matrix[i].non_zeros;
    }
    rows = malloc(non_zeros*sizeof(int));
    columns = malloc(non_zeros*sizeof(int));
    values = malloc(non_zeros*sizeof(int));
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < matrix[i].non_zeros; ++j){
            rows[count] = i;
            columns[count] = matrix[i].columns[j];
            values[count] = matrix[i].values[j];
            count++;
        }
    }
}

//trying to save as much memory as possible
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
    int mm_non_zeros,CLUSTERS = 0, ROW_COUNT,COL,*c, *rowsA,*colsA, *valA, dest_r, dest_col,file_non_zeros;
    double read_temp;
    char *matrixFilePath, *clusterVectorPath;
    if(argc == 3){
        //use user defined paths from command line
        matrixFilePath = argv[1];
        clusterVectorPath = argv[2];
    } 
    else{
        //use our default
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
        mm_non_zeros = 2*file_non_zeros; //make it double temporarily, because file contains only elements of below triangle of matrix
    }
    else{
        mm_non_zeros = file_non_zeros;
    }
    c = malloc(ROW_COUNT*sizeof(int));
    rowsA = malloc(mm_non_zeros*sizeof(int));
    colsA = malloc(mm_non_zeros*sizeof(int));
    valA = malloc(mm_non_zeros*sizeof(int));
    memset(c,0, ROW_COUNT*sizeof(int)); //just in case
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
            //accept only positive weights
            if (read_temp < 0) read_temp = -read_temp;
            if ( (int) read_temp == 0){
                //test
                read_temp = 1;
            }
            valA[i] = (int) read_temp;
        }
        if(mm_is_symmetric(matcode)){
            if(rowsA[i] != colsA[i]){
                rowsA[file_non_zeros+counter] = colsA[i];
                colsA[file_non_zeros+counter] = rowsA[i];
                valA[file_non_zeros+counter] = valA[i];
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
                counter++;
                actual_non_zeros += 2;
            }
            else{ //diagonal element
                actual_non_zeros++;
            }
        }
    }
    fclose(f);
    if(mm_is_symmetric(matcode) || mm_is_skew(matcode)){
        mm_non_zeros = actual_non_zeros;
        //save memory
        rowsA = realloc(rowsA, mm_non_zeros*sizeof(int));
        colsA = realloc(colsA, mm_non_zeros*sizeof(int));
        valA = realloc(valA,mm_non_zeros*sizeof(int));
    }
    /*int r[] = {1,1,2,2,3}, co[] = {1,2,1,3,1}, v[] = {1,1,1,1,1}; //represents mm file
    int c[] = {1,1,2};*/
    struct row *minor = malloc(CLUSTERS*sizeof(struct row));
    for(int i = 0; i<CLUSTERS;++i){
        minor[i].non_zeros = 0;
    }
    bool stored = false;
    int pos=0;
    struct timeval start,end;
    gettimeofday(&start,0);
    for(int i = 0; i < mm_non_zeros; ++i){
        stored = false;
        dest_r = c[rowsA[i]-1] -1;
        dest_col = c[colsA[i]-1]-1;
        if(minor[dest_r].non_zeros == 0){
            saveResult(&minor[dest_r], dest_col, valA[i]);
            stored = true;
        }
        else{
            for(int j = 0; j < minor[dest_r].non_zeros;++j){
                if(minor[dest_r].columns[j] == dest_col){
                    minor[dest_r].values[j] += valA[i];
                    stored = true;
                }
            }
        }
        if(!stored){ //new column found
            saveResult(&minor[dest_r], dest_col,valA[i]);
        }
    }
    gettimeofday(&end,0);
    long secondsElapsed = end.tv_sec - start.tv_sec;
    long microsecondsElapsed = end.tv_usec - start.tv_usec;
    double totalTime = secondsElapsed + 1e-6*microsecondsElapsed;
    printf("-------------\nminor matrix:\n");
    printrows(minor,CLUSTERS);
    printf("Processing time: %f seconds\n",totalTime);
    //cleanup
    free(rowsA);
    free(colsA);
    free(valA);
    /*
    In case of converting back to COO, it can be done like this. It's just a memory rearrangement.

    int *rowsMinor, *colsMinor, *valuesMinor;
    CSCtoCOO(minor, CLUSTERS, rowsMinor, colsMinor, valuesMinor);
    
    */
    for(int i = 0; i < CLUSTERS;++i){
        free(minor[i].columns);
        free(minor[i].values);
    }
    free(minor);
    return 0;
}