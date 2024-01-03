#include <stdio.h>
#include <stdlib.h>
#include "include/utils.h"

int main(int argc, char* argv[]){
    size_t k = 5, elements = 6;
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
    uint32_t* vec = malloc(elements*sizeof(uint32_t));
    double read_temp;
    for(int i = 0; i < elements;++i){
        fscanf(file, "%lg", &read_temp);
        vec[i] = (uint32_t) read_temp;
    }
    fclose(file);
    uint32_t value = quickselect(vec, 0, elements-1, k);
    printf("The %zu th smallest number is %u\n", k, value);
    free(vec);
    return 0;
}
