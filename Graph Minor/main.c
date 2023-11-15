#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "mmio.h"
#include <string.h>
#include <sys/time.h>

//it may be slow and clunky, I know
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

int main(void){
    int mm_non_zeros,CLUSTERS = 0, ROW_COUNT,COL,*c, *rowsA,*colsA, *valA, dest_r, dest_col;
    double read_temp;
    MM_typecode matcode;
    FILE* f = fopen("data/dwt_2680.mtx","r");
    FILE* c_f = fopen("data/out10-dwt.txt","r");
    if(f == NULL || c_f == NULL){
        printf("Error! input files not found!\n");
        exit(1);
    }
    mm_read_banner(f,&matcode);
    mm_read_mtx_crd_size(f,&ROW_COUNT,&COL,&mm_non_zeros);
    c = malloc(ROW_COUNT*sizeof(int));
    rowsA = malloc(mm_non_zeros*sizeof(int));
    colsA = malloc(mm_non_zeros*sizeof(int));
    valA = malloc(mm_non_zeros*sizeof(int));
    //memset(c,0, ROW_COUNT*sizeof(int)); //just in case
    for(int i = 0; i < ROW_COUNT;++i){
        fscanf(c_f,"%lg",&read_temp);
        c[i] = (int) read_temp;
        if(c[i] > CLUSTERS){
            CLUSTERS = c[i];
        }
    }
    fclose(c_f);
    for(int i = 0; i < mm_non_zeros; ++i){
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
        //printf("%d %d %lg\n", rowsA[i], colsA[i],valA[i]);
    }
    fclose(f);
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
    //this is fast AF boi
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
    for(int i = 0; i < CLUSTERS;++i){
        free(minor[i].columns);
        free(minor[i].values);
    }
    free(minor);
}