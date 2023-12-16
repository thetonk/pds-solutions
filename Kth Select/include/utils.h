#ifndef UTILS_H_
#define UTILS_H_

void swap(int *a, int *b);

void printArray(int *array, int n);
void printArray2(int *array, int l, int r);

int partition(int *array, int l, int r);

//int partitionMPI(int root, int* array, int l, int r);

//int partitionMPI2(int root, int* array, int l, int r);

int quickselect(int *array, int l, int r, int k);

//int quickselectMPI(int root, int *array, int l, int r ,int k);

int quickselectMPI2(int root, int *array, int arraySize, int l, int r, int k);

#endif // UTILS_H_
