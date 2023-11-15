#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "mmio.h"
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

#define MAX_THREADS 5

pthread_t threads[MAX_THREADS];
pthread_mutex_t myMutex = PTHREAD_MUTEX_INITIALIZER;

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

void* graphMinor(void* arg){
    struct ThreadData* myData = (struct ThreadData*) arg; 
    int start = (NON_ZEROS/MAX_THREADS)*(myData->threadID);
    int stop  = (myData->threadID == MAX_THREADS - 1 )? NON_ZEROS: start + NON_ZEROS/MAX_THREADS;
    int dest_row, dest_col, non_zeros;
    bool stored = false;
    for(int i = start; i < stop; ++i){
        stored = false;
        dest_row = c[rowsA[i]-1] -1;
        dest_col = c[colsA[i]-1]-1;
        pthread_mutex_lock(&myMutex);
        non_zeros = minor[dest_row].non_zeros;
        pthread_mutex_unlock(&myMutex);
        if(non_zeros == 0){
            pthread_mutex_lock(&myMutex);
            saveResult(&minor[dest_row], dest_col, valA[i]);
            pthread_mutex_unlock(&myMutex);
            stored = true;
            
        }
        else{
            for(int j = 0; j < non_zeros;++j){
                if(minor[dest_row].columns[j] == dest_col){
                    pthread_mutex_lock(&myMutex);
                    minor[dest_row].values[j] += valA[i];
                    pthread_mutex_unlock(&myMutex);
                    stored = true;
                }
            }
        }
        if(!stored){ //new column found
            pthread_mutex_lock(&myMutex);
            saveResult(&minor[dest_row], dest_col,valA[i]);
            pthread_mutex_unlock(&myMutex);
        }
    }
    return (NULL);
}

void* graphMinorRe(void *arg){
    struct ThreadData *myData = (struct ThreadData*) arg;
    int dest_row, dest_col, non_zeros,index;
    bool stored = false;
    for(int i = 0 ; i < myData->non_zeros; ++i){
        stored = false;
        index = myData->non_zeros_indexes[i];
        dest_row = c[rowsA[index]-1] -1;
        dest_col = c[colsA[index]-1]-1;
        non_zeros = minor[dest_row].non_zeros;
        if(non_zeros == 0){
            saveResult(&minor[dest_row], dest_col, valA[index]);
            stored = true;
            
        }
        else{
            for(int j = 0; j < non_zeros;++j){
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
    return (NULL);
}

int main(void){
    int COL,ROW_COUNT, NON_ZEROS_PER_THREAD[MAX_THREADS];
    double read_temp;
    MM_typecode matcode;
    FILE* f = fopen("data/dwt_2680.mtx","r");
    FILE* c_f = fopen("data/out10-dwt.txt","r");
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
            if((c[rowsA[j] -1 ] -1) % MAX_THREADS == i){
                threadData[i].non_zeros_indexes[counter] = j;
                counter++;
            }
        }
    }
    //end overhead
    gettimeofday(&start,0);
    for(int i = 0; i < MAX_THREADS; ++i){
        pthread_create(&threads[i],NULL, graphMinorRe, &threadData[i]);
    }
    for(int i = 0; i < MAX_THREADS; ++i){
        pthread_join(threads[i],NULL);
    }
    gettimeofday(&end,0);
    printf("-------------\nminor matrix:\n");
    printrows(minor,CLUSTERS);
    long secondsElapsed = end.tv_sec - start.tv_sec;
    long microsecondsElapsed = end.tv_usec - start.tv_usec;
    double totalTime = secondsElapsed + 1e-6*microsecondsElapsed;
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
    pthread_mutex_destroy(&myMutex);
    return 0;
}