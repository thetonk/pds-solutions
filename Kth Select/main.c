#include <stdio.h>
#include <stdlib.h>
#include "include/utils.h"

int main(int argc, char* argv[]){
    int k = 5, elements = 6;
    char *inputFilePath;
    if (argc < 4) {
        printf("Invalid argument count! Exiting!\n");
        printf("[USAGE] ./test-openmpi <input file path> <number of elements> <element position to search>\n");
        return 1;
    }
    else{
        inputFilePath = argv[1];
        elements = atoi(argv[2]);
        k = atoi(argv[3]) - 1;
        if(k < 0 || k > elements - 1){
            printf("Error! invalid k-th value. K-th value muse be positive and smaller or equal to the amount of elements.\n");
            return 2;
        }
    }
    FILE* file = fopen(inputFilePath,"r");
    if (file == NULL){
        printf("Error! file not found!");
        return 3;
    }
    int* vec = malloc(elements*sizeof(int));
    double read_temp;
    for(int i = 0; i < elements;++i){
        fscanf(file, "%lg", &read_temp);
        vec[i] = (int) read_temp;
    }
    fclose(file);
    int value = quickselect(vec, 0, elements-1, k);
    printf("The %d th smallest number is %d\n", k, value);
    free(vec);
    return 0;
}
