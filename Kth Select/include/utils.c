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

//using Hoare's partitioning, to make as few swaps as possible
int partitionMPI(int root, int *array, int l, int r){
    //printf("l is %d r is %d\n", l ,r);
    int pivot,position,rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i = l-1, j = r+1;
    if(rank == root){
        pivot = array[l+(r-l)/2];
    }
    //broadcast pivot
    MPI_Bcast(&pivot, 1, MPI_INT, root, MPI_COMM_WORLD);
    printf("pivot is %d! from process rank %d\n", pivot,rank);
    while(true && i <= r && j >= l){
        while(array[++i] < pivot);
        while(array[--j] > pivot);
        //printf("i is %d j is %d from process rank %d\n", i,j,rank);
        //when pointers meet, this is the pivot's correct position
        if(i>=j){
            position = j;
            break;
        }
        swap(&array[i], &array[j]);
    }
    //if(rank == root) position++;
    printf("rank %d pivot local position is %d, with value: %d\n",rank,position,array[position]);
    return position;
}

//using Lemuto's partitioning
int partitionMPI2(int root, int *array, int l, int r){
    //printf("l is %d r is %d\n", l ,r);
    int pivot,position,rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i = l-1;
    if(rank == root){
        //pick last element
        pivot = array[r];
    }
    //broadcast pivot
    MPI_Bcast(&pivot, 1, MPI_INT, root, MPI_COMM_WORLD);
    printf("pivot is %d! from process rank %d\n", pivot,rank);
    for(int j = l; j < r; ++j){
        if(array[j] <= pivot){
            i++;
            swap(&array[i], &array[j]);
        }
    }
    i++;
    swap(&array[r], &array[i]);
    position = i;
    printf("rank %d pivot local position is %d, with value: %d\n",rank,position,array[position]);
    return position;
}

int quickselectMPI(int root, int *array, int l, int r, int k){
    int pivot_local, pivot_global, rank,n_procs;
    bool termination = false;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    int localPivots[n_procs];
    if(r >= l){
        pivot_local = partitionMPI2(root,array,l,r);
        printArray2(array, l,r);
        MPI_Allreduce(&pivot_local, &pivot_global, 1, MPI_INT, MPI_SUM,MPI_COMM_WORLD);
        MPI_Allgather(&array[pivot_local], 1, MPI_INT, localPivots,1, MPI_INT, MPI_COMM_WORLD);
        printf("rank %d global pivot is %d\n", rank,pivot_global);
        printf("rank %d gathered local pivots: ",rank );
        printArray(localPivots, n_procs);
        if(r == l && (k >= pivot_global && k <= pivot_global + n_procs)){
            //MPI_Barrier(MPI_COMM_WORLD);
            if(termination && r==l){
                return array[l];
            }
            else{
                //the selected value is one of the pivots, find it with quickselect
                int value = quickselect(localPivots, 0, n_procs-1, k - pivot_global);
                printf("value is %d!\n", value);
                termination = true;
                MPI_Bcast(&termination, 1, MPI_C_BOOL, rank, MPI_COMM_WORLD);
                return value;
            }
        }
        else if (pivot_global > k ){
            //search left
            printf("rank %d moving LEFT!\n", rank);
            return quickselectMPI(root, array, l, pivot_local-1,k);
        }
        else{
            //search right
            printf("rank %d moving RIGHT!\n", rank);
            return quickselectMPI(root, array, pivot_local+1, r, k);
        }
    }
    else{
        printf("error on rank %d. l is %d, r is %d\n",rank, l ,r);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }
}

int quickselectMPI2(int root, int *array, int l, int r, int k){
    int pivot, position, rank, smallerCount=0, pivotDupes=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int global_min, global_max, global_position, global_pivotDupes;
    int local_min = INT_MAX, local_max = INT_MIN;
    if (r >= l){
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
    else{
        if (rank != root && r < 0) local_max = array[l];
        if (rank != root && l > r) local_min = array[r];
    }
    printf("rank %d; r = %d, l = %d\n", rank,r, l);
    //subarray overflow
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min,1,MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == root){
        //get minimum of other subarrays non eliminated elements, if going right, else get maxium if going left
        if ( r < 0){
            pivot = global_max;
        }
        else if (l > r){
            pivot = global_min;
        }
    }
    printf("rank %d; r = %d, l = %d\n", rank,r, l);
    //subarray overflow
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min,1,MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Bcast(&pivot, 1, MPI_INT, root, MPI_COMM_WORLD);
    printf("rank %d global min: %d, global max: %d, pivot: %d\n", rank,global_min, global_max, pivot);
    //partition using lemuto's method
    int i = -1;
    for(int j = 0; j < r ;++j){
        if(array[j] <= pivot){
            i++;
            swap(&array[i], &array[j]);
            if (array[j] < pivot) smallerCount++;
            if(array[j] == pivot) pivotDupes++;
        }
    }
    i++;
    if( r >= 0 && array[r] == pivot) pivotDupes++;
    if(r >= 0 && array[r] < pivot) smallerCount++;
    swap(&array[r], &array[i]);
    position = i;
    //sum local positions to find the global one
    MPI_Allreduce(&smallerCount, &global_position, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pivotDupes, &global_pivotDupes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("[RANK %d] pivot value is %d, global position is %d and pivot dupes are %d\n", rank, pivot, global_position,global_pivotDupes);
    if(k >= global_position && k < global_position + global_pivotDupes ){
        //found the kth number
        printf("rank %d found k!\n", rank);
        return pivot;
    }
    // otherwise, start splitting, oh boy, here comes the "sun" :( ...
    if (k < global_position){
        return quickselectMPI2(root, array, l,  smallerCount-1, k);
    }
    else{
        if(rank == root) return quickselectMPI2(root, array, smallerCount+1, r,k);
        return quickselectMPI2(root, array, smallerCount+1, r, k);
    }
}
