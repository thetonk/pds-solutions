#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

//create custom size_t MPI datatype since it does not exist
#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "Couldn't find the size of biggest datatype in this system. What the hell?"
#endif

void swap(uint32_t *a, uint32_t *b);

void printCommandUsage();

void printArray(uint32_t *array, size_t n);

void printArray2(uint32_t *array, int64_t l, int64_t r);

int64_t partition(uint32_t *array, int64_t l, int64_t r);

uint32_t quickselect(uint32_t *array, int64_t l, int64_t r, size_t k);

uint32_t quickselectMPI2(int root, uint32_t *array, size_t arraySize, int64_t l, int64_t r, size_t k);

#endif // UTILS_H_
