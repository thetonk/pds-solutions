#include "utils.h"
#include <stdio.h>
#include <mpi.h>
#include <limits.h>
#include <stdbool.h>

void printArray(int *array, int n){
    printf("[");
    for(int i = 0; i < n; ++i){
        printf("%d ", array[i]);
    }
    printf("]\n");
}

void printArray2(int *array, int l, int r){
    printf("[");
    for(int i = l; i <=r; ++i){
        printf("%d ", array[i]);
    }
    printf("]\n");
}

void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

//using Hoare's partitioning, to make as few swaps as possible
int partition(int *array, int l, int r){
    //select middle element as pivot
    int pivot = array[l+(r-l)/2], position;
    int i = l-1, j = r+1;
    while(true){
        while(array[++i] < pivot);
        while(array[--j] > pivot);
        //when pointers meet, this is the pivot's correct position
        if(i>=j){
            position = j;
            break;
        }
        swap(&array[i], &array[j]);
    }
    return position;
}


int quickselect(int* array, int l, int r, int k){
    if(l == r) return array[l];
    int p = partition(array, l, r);
    if (k == p) return array[p];
    //search on left sub array if k is smaller than pivot position
    else if(k < p){
        return quickselect(array, l, p-1, k);
    }
    //else search on the right
    else{
        return quickselect(array, p+1, r, k);
    }
}

int quickselectMPI2(int root, int *array, int arraySize, int l, int r, int k){
    int pivot, position, rank, smallerCount=0, pivotDupes=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int global_min, global_max, global_position, global_pivotDupes;
    int local_min = INT_MAX, local_max = INT_MIN;
    if (r >= l){
        printf("rank %d: ", rank);
        printArray2(array, l, r);
        //get min and max of other subarrays
        //local_max = array[l], local_min = array[l];
        for(int i = l; i <= r; ++i){
            if (array[i] > local_max){
                local_max = array[i];
            }
            if(array[i] < local_min){
                local_min = array[i];
            }
        }
        if(rank == root){
            pivot = array[r];
        }
    }
    printf("rank %d; r = %d, l = %d\n", rank,r, l);
    //subarray overflow
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min,1,MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == root){
        //get minimum of other subarrays non eliminated elements, if going right, else get maxium if going left
        if ( r < 0 || l > r ){
            pivot = global_min;
        }
    }
    //subarray info
    MPI_Bcast(&pivot, 1, MPI_INT, root, MPI_COMM_WORLD);
    printf("rank %d global min: %d, global max: %d, pivot: %d\n", rank,global_min, global_max, pivot);
    //partition using lemuto's method
    int i = -1;
    for(int j = 0; j < arraySize -1 ;++j){
        if(array[j] <= pivot){
            i++;
            swap(&array[i], &array[j]);
        }
    }
    i++;
    swap(&array[arraySize -1], &array[i]);
    position = i;
    for(int j = 0; j < arraySize; ++j){
        if (array[j] < pivot) smallerCount++;
        if(array[j] == pivot) pivotDupes++;
    }
    //sum local positions to find the global one
    MPI_Allreduce(&smallerCount, &global_position, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pivotDupes, &global_pivotDupes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("[RANK %d] pivot value is %d, global position is %d, smaller count is %d and pivot dupes are %d\n", rank, pivot, global_position,smallerCount,global_pivotDupes);
    if(k >= global_position && k < global_position + global_pivotDupes ){
        //found the kth number
        printf("rank %d found k!\n", rank);
        return pivot;
    }
    // otherwise, start splitting, oh boy, here comes the "sun" :( ...
    if (k < global_position){
        if(array[smallerCount] == pivot) return quickselectMPI2(root, array, arraySize, l,  smallerCount-pivotDupes, k);
        if(rank != root && smallerCount - l > r - l){
            // #HACK out of bounds
            printf("rank %d: REBOUNDING!\n", rank);
            return quickselectMPI2(root, array, arraySize, l, r, k);
        }
        return quickselectMPI2(root, array, arraySize, l, smallerCount, k);
    }
    else{
        if(array[smallerCount] == pivot) return quickselectMPI2(root, array, arraySize,smallerCount+pivotDupes, r,k);
        if(rank != root && r - smallerCount > r - l){
            // #HACK out of bounds
            printf("rank %d: REBOUNDING!\n", rank);
            return quickselectMPI2(root, array,arraySize, l, r, k);
        }
        return quickselectMPI2(root, array, arraySize,smallerCount, r, k);
    }
}
