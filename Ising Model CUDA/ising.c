#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#define N 10

void printLattice(int8_t *lattice, size_t n){
    for(size_t row= 0; row <n; ++row){
        for(size_t col = 0; col < n ; ++col){
            printf("%2d ", lattice[n*row + col]);
        }
        putchar('\n');
    }
}

void generateRandomLattice(int8_t *lattice, size_t n){
    for(size_t row = 0; row < n; ++row){
        for(size_t col = 0; col < n; ++col){
            lattice[n*row + col] = 2*(rand() % 2) - 1;
        }
    }
}

int8_t getIndex(int8_t i, size_t n){
    return i == -1 ? n-1 : i % n;
}

void calculateNextLattice(int8_t *curLattice, int8_t *nexLattice,size_t n){
    int8_t sum;
    for(size_t row = 0; row < n; ++row){
        for(size_t col = 0; col < n; ++col){
            sum = curLattice[n*row + col] + curLattice[n*getIndex(row-1, n) + col] + curLattice[n*getIndex(row+1, n) + col] +
                curLattice[n*row + getIndex(col -1, n)] +curLattice[n*row + getIndex(col+1, n)];
            //calculate sign
            nexLattice[n*row + col] = (sum >= 0)? 1 : -1;
        }
    }
}

int main(int argc, char *argv[]){
    //row major order will be followed
    size_t elementsPerRow = N, epochs = 5, seed = 69;
    int8_t *current_lattice_state, *next_lattice_state, *temp;
    //argument parsing
    switch(argc){
        case 4:
            elementsPerRow = atoi(argv[1]);
            epochs = atoi(argv[2]);
            seed = atoi(argv[3]);
            break;
        case 3:
            elementsPerRow = atoi(argv[1]);
            epochs = atoi(argv[3]);
            break;
        case 2:
            elementsPerRow = atoi(argv[1]);
            break;
    }
    //assume square lattice
    current_lattice_state = malloc(elementsPerRow*elementsPerRow*sizeof(int8_t));
    next_lattice_state = malloc(elementsPerRow*elementsPerRow*sizeof(int8_t));
    srand(seed);
    generateRandomLattice(current_lattice_state, elementsPerRow);
    printLattice(current_lattice_state, elementsPerRow);
    sleep(5);
    for(size_t i = 0; i < epochs; ++i){
        printf("=========================================================\n");
        calculateNextLattice(current_lattice_state, next_lattice_state, elementsPerRow);
        //lazily set next lattice state as the new one
        temp = current_lattice_state;
        current_lattice_state = next_lattice_state;
        next_lattice_state = temp;
        system("clear");
        printLattice(current_lattice_state, elementsPerRow);
        sleep(1);
    }
    free(current_lattice_state);
    free(next_lattice_state);
    return 0;
}
