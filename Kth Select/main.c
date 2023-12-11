#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

//using Hoare's partitioning, to make as few swaps as possible
int partition(int *array, int l, int r){
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
    //if(l == r) return array[l];
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

int main(int argc, char* argv[]){
    int k = 3;
    int vec[] = {3, 10 ,2,1,54,3};
    int value = quickselect(vec, 0, 5, k);
    printf("The %d th smallest number is %d\n", k, value);
    return 0;
}
