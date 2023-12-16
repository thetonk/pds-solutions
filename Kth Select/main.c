#include <stdio.h>
#include <stdlib.h>
#include "include/utils.h"

int main(int argc, char* argv[]){
    int k = 5, elements = 6;
    FILE* file = fopen("data/input.txt", "r");
    if (file == NULL){
        printf("Error! file not found!");
        exit(1);
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
