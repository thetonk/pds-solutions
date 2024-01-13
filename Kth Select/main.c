#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "include/utils.h"
#include "include/curlutils.h"

int main(int argc, char* argv[]){
    size_t k = 5, elements = 6;
    bool isFile;
    char *inputFilePath, *URL;
    uint32_t *vec, value;
    //timing utilities
    struct timeval start, end;
    long secondsElapsed, microSecondsElapsed;
    double totalTime;

    //argument parsing
    if (argc < 4) {
        printf("Invalid argument count! Exiting!\n");
        printCommandUsage("test");
        return 1;
    }
    else{
        if(argv[1][0] == '-'){
            switch(argv[1][1]){
                case 'f': //file mode
                    isFile = true;
                    inputFilePath = argv[2];
                    elements = atoi(argv[3]);
                    k = atoi(argv[4]) - 1;
                    break;
                case 'l': //URL mode
                    isFile = false;
                    URL = argv[2];
                    k = atoi(argv[3]) - 1;
                    break;
                default: //invalid parameter
                    printf("Invalid parameter! Exiting!\n");
                    printCommandUsage("test-openmpi");
                    return 1;
            }
        }
        else{
            if(argc == 4){
                isFile = true;
                inputFilePath = argv[1];
                elements = atoi(argv[2]);
                k = atoi(argv[3]) - 1;
            }
            else{
                 printf("Invalid parameter! Exiting!\n");
                 printCommandUsage("test-openmpi");
                 return 1;
            }
        }
        if(k < 0){
            printf("Error! invalid k-th value. K-th value muse be positive and smaller or equal to the amount of elements.\n");
            return 2;
        }
    }
    if(isFile){
        FILE* file = fopen(inputFilePath,"r");
        if (file == NULL){
            printf("Error! file not found!");
            return 3;
        }
        vec = malloc(elements*sizeof(uint32_t));
        double read_temp;
        for(int i = 0; i < elements;++i){
            fscanf(file, "%lg", &read_temp);
            vec[i] = (uint32_t) read_temp;
        }
        fclose(file);
        gettimeofday(&start, 0);
        value = quickselect(vec, 0, elements-1, k);
        gettimeofday(&end,0);
        free(vec);
    }
    else{
        ARRAY myData = getWikiFull(URL);
        gettimeofday(&start, 0);
        value = quickselect(myData.data, 0, myData.size-1,k);
        gettimeofday(&end,0 );
        free(myData.data);
    }
    secondsElapsed = end.tv_sec - start.tv_sec;
    microSecondsElapsed = end.tv_usec - start.tv_usec;
    totalTime = secondsElapsed + 1e-6*microSecondsElapsed;
    printf("The %zu th smallest number is %u\n", k+1, value);
    printf("Time elapsed: %f seconds\n", totalTime);
    return 0;
}
