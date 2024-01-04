#include "utils.h"
#include <mpi.h>
#include <stdio.h>

void printCommandUsage(){
    printf("[USAGE] ./test-openmpi <input file path> <number of elements> <element position to search>\n");
    printf("[USAGE] ./test-openmpi -l <URL> <element position to search>\n");
    printf("[USAGE] ./test-openmpi -f <input file path> <number of elements> <element position to search>\n");
}

void printArray(uint32_t *array, size_t n){
    printf("[");
    for(size_t i = 0; i < n; ++i){
        printf("%u ", array[i]);
    }
    printf("]\n");
}

void printArray2(uint32_t *array, int64_t l, int64_t r){
    printf("[");
    for(int64_t i = l; i <=r; ++i){
        printf("%u ", array[i]);
    }
    printf("]\n");
}

void swap(uint32_t *a, uint32_t *b){
    uint32_t temp = *a;
    *a = *b;
    *b = temp;
}

//using Hoare's partitioning, to make as few swaps as possible
int64_t partition(uint32_t *array, int64_t l, int64_t r){
    //select middle element as pivot
    uint32_t pivot = array[l+(r-l)/2], position;
    int64_t i = l-1, j = r+1;
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


uint32_t quickselect(uint32_t* array, int64_t l, int64_t r, size_t k){
    if(l == r) return array[l];
    uint32_t p = partition(array, l, r);
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

uint32_t quickselectMPI2(int root, uint32_t *array, size_t arraySize, int64_t l, int64_t r, size_t k){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    uint32_t global_min, global_max,pivot;
    size_t global_position, global_pivotDupes,position, smallerCount=0, pivotDupes=0;
    uint32_t local_min = UINT32_MAX, local_max = 0;
    if (l >-1 && r >= l){
        printf("rank %d: ", rank);
        if(r-l < 20) printArray2(array, l, r);
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
    printf("rank %d; r = %ld, l = %ld\n", rank,r, l);
    //subarray overflow
    //MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); not needed anymore
    MPI_Allreduce(&local_min, &global_min,1,MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    if (rank == root){
        //get minimum of other subarrays non eliminated elements, if going right, else get maxium if going left
        if (r < 0 || l > r ){
            pivot = global_min;
        }
    }
    //subarray info
    MPI_Bcast(&pivot, 1, MPI_UINT32_T, root, MPI_COMM_WORLD);
    printf("rank %d global min: %u, global max: %u, pivot: %u\n", rank,global_min, global_max, pivot);
    //partition using lomuto's method
    int64_t i = -1;
    for(size_t j = 0; j < arraySize -1 ;++j){
        if(array[j] <= pivot){
            i++;
            swap(&array[i], &array[j]);
        }
    }
    i++;
    swap(&array[arraySize -1], &array[i]);
    position = i;
    for(size_t j = 0; j < arraySize; ++j){
        if (array[j] < pivot) smallerCount++;
        if(array[j] == pivot) pivotDupes++;
    }
    //sum local positions to find the global one
    MPI_Allreduce(&smallerCount, &global_position, 1, my_MPI_SIZE_T , MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pivotDupes, &global_pivotDupes, 1, my_MPI_SIZE_T , MPI_SUM, MPI_COMM_WORLD);
    printf("[RANK %d] local position is %zu\n", rank,position);
    printf("[RANK %d] pivot value is %u, global position is %zu, smaller count is %zu and pivot dupes are %zu\n", rank, pivot, global_position,smallerCount,global_pivotDupes);
    if(k >= global_position && k < global_position + global_pivotDupes ){
        //found the kth number
        printf("rank %d found k!\n", rank);
        return pivot;
    }
    // otherwise, start splitting, oh boy, here comes the "sun" :( ...
    if (k < global_position){
        if(rank == root){
            //pivot is the smallest element, elements of root array are depleted
            if(smallerCount == 0) return quickselectMPI2(root, array, arraySize, l, -1, k);
            //pivot is the largest element, elements of root array are depleted
            //else if (smallerCount == (r-l)+1) return quickselectMPI2(root, array, arraySize, r+1, r, k);
            else{
                int64_t p;
                //select one element not equal to pivot, otherwise it will get stuck
                for(p = r; p >= l; p--){
                    if(array[p] < pivot) break;
                }
                return quickselectMPI2(root, array, arraySize, l, p,k);
            }
        }
        else{
            if( ((int64_t) position) - l > r - l){
                printf("rank %d: REBOUNDING!\n", rank);
                return quickselectMPI2(root, array, arraySize, l, r, k);
            }
            if(array[position] == pivot) return quickselectMPI2(root, array, arraySize, l, position-1, k);
            return quickselectMPI2(root, array, arraySize, l, position, k);
        }
    }
    else{
        if(rank == root){
            //pivot is the smallest element, elements of root array are depleted
            //if(smallerCount == 0) return quickselectMPI2(root, array, arraySize, l, -1, k);
            //pivot is the largest element, elements of root array are depleted
            if (smallerCount == arraySize-1) return quickselectMPI2(root, array, arraySize, r+1, r, k);
            else{
                int64_t p;
                //select one element not equal to pivot, otherwise it will get stuck
                for(p = l; p <= r; p++){
                    if(array[p] > pivot) break;
                }
                return quickselectMPI2(root, array, arraySize, p, r,k);
            }
        }
        else{
            if(r - ((int64_t) position) > r - l){
                printf("rank %d: REBOUNDING!\n", rank);
                return quickselectMPI2(root, array,arraySize, l, r, k);
            }
            if(array[position] == pivot) return quickselectMPI2(root, array, arraySize, position+1, r, k);
            return quickselectMPI2(root, array, arraySize,position, r, k);
        }
    }
}
