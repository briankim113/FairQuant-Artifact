/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "split.h"



int main( int argc, char *argv[])
{
    char *FULL_NET_PATH;

    if (argc > 3) {
        fprintf(stderr, "Number of arguments given: %d\n", argc);
        fprintf(stderr, "The correct format is\n");
        fprintf(stderr, "\t./network_test [network] [sens_feature_idx]\n");
        exit(1);
    }

    for (int i=1;i<argc;i++) {
        if (i == 1) {
            FULL_NET_PATH = argv[i];
        }

        if (i == 2) {
            SENS_FEATURE_IDX = atoi(argv[i]);
        }
    }

    openblas_set_num_threads(1);
    srand((unsigned)time(NULL));

    int i,j,layer;

    struct NNet* nnet = load_network(FULL_NET_PATH, SENS_FEATURE_IDX);
    load_positive_and_negative_weights(nnet);

    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;
    
    //input
    float input_lower[inputSize], input_upper[inputSize]; //contains our subproblem
    struct Interval input_interval = {
        (struct Matrix){input_lower, 1, inputSize},
        (struct Matrix){input_upper, 1, inputSize},
    };

    unsigned long long global_volume = 1; // number of possible male (or female) individuals, so total universe is this number * 2

    for (int i=0; i<inputSize; i++){
        if (i == nnet->sens_feature_idx){
            input_lower[i] = input_upper[i] = 0; //set our gender inside later
        }
        else {
            input_lower[i] = nnet->mins[i]; // initially read nnet input ranges
            input_upper[i] = nnet->maxes[i];
            global_volume *= ((int) nnet->ranges[i]) + 1; //number of distinct int values that a feature can take
        }
    }

    nnet->global_volume = global_volume; // number of possible individuals, considering values can only take ints

    int n = 0;

    for (int i=0; i<inputSize; i++) {
        if (input_interval.upper_matrix.data[i] < input_interval.lower_matrix.data[i]) {
            fprintf( stderr, "wrong input!\n");
            exit(1);
        }
        if (input_interval.upper_matrix.data[i] != input_interval.lower_matrix.data[i]){
            n++;
        }
    }

    int feature_range_length = n;
    int *feature_range = (int*)malloc(n*sizeof(int));

    for (int i=0, n=0; i<inputSize; i++) {
        if (input_interval.upper_matrix.data[i] != input_interval.lower_matrix.data[i]){
            feature_range[n] = i;
            n++;
        }
    }
    
    // these variables are for timeout and memoryout: important to stay consistent for experiment configs
    long int TIMEOUT = 1800; //in seconds
    int MAX_SPLIT_DEPTH = 20; //allow splitting intervals up to depth
    int MIN_CHECK_DEPTH = 15; //allow checking advs from this depth
    int PQ_MAXSIZE = MAX_SPLIT_DEPTH*2; //max can be the depth of the tree, allowing *2 to be safe


    struct Interval *pq_intervals = malloc(PQ_MAXSIZE*sizeof(struct Interval));             // allocate memory to our intervals to store, one interval per subproblem
    struct Subproblem *pq_subproblems = malloc(PQ_MAXSIZE*sizeof(struct Subproblem));       // allocate memory to our subproblems to store, two subproblems per split
    float input_uppers[PQ_MAXSIZE][inputSize], input_lowers[PQ_MAXSIZE][inputSize];         // where the data are stored, referenced by intervals and subproblems

    int subproblem_remaining = 0;
    int subproblem_total = 0;
    double time_spent = 0;

    // insert interval into array 
    pq_intervals[subproblem_remaining] = input_interval;
    
    // insert subproblem into array
    gettimeofday(&curr, NULL); time_spent = ((float)(curr.tv_sec - start.tv_sec) * 1000000 + (float)(curr.tv_usec - start.tv_usec)) / 1000000;
    pq_subproblems[subproblem_remaining] = (struct Subproblem){pq_intervals[subproblem_remaining], 0, time_spent};

    // increase counters
    subproblem_remaining++;
    subproblem_total++;

    // for calculations
    unsigned long long cert_volume = 0;
    unsigned long long fals_volume = 0;
    unsigned long long adv_volume = 0;     // entire volume of intervals that contain adversarial examples
    unsigned long long num_adv = 0;        // number of actual adversarial examples
    unsigned long long uncer_volume = nnet->global_volume;

    // start of while loop
    struct Subproblem curr_subp;
    gettimeofday(&start, NULL);

    fprintf(stdout, "\nrunning network %s on sens_feature_idx = %d\n\n", FULL_NET_PATH, SENS_FEATURE_IDX);

    while (1){

        if (subproblem_remaining <= 0) {
            fprintf( stdout, "no more subproblems, complete termination\n" );
            break;
        }

        if (uncer_volume < 0.00001*nnet->global_volume) {
            fprintf( stdout, "very low rateUncer < 0.00001, quasi-complete termination\n" );
            break;
        }

        if (subproblem_remaining >= PQ_MAXSIZE){
            fprintf( stderr, "memoryout of %d subps remaining: exiting before SIGKILL called\n", subproblem_remaining );
            break;
        }

        gettimeofday(&curr, NULL);
        if (curr.tv_sec - start.tv_sec > TIMEOUT){
            fprintf( stderr, "timeout of %ld seconds\n", TIMEOUT );
            break;
        }

        // depth_first_search
        subproblem_remaining--; // we are now looking at this subproblem
        curr_subp = pq_subproblems[subproblem_remaining];
        unsigned long long curr_volume = 1;

        // set up input0, input1
        float i0_lower[inputSize], i0_upper[inputSize];
        float i1_lower[inputSize], i1_upper[inputSize];

        for (int i=0; i<inputSize; i++){
            // how many individuals are represented by this subp
            if (i == nnet->sens_feature_idx){
                i0_lower[i] = i0_upper[i] = nnet->mins[i];  //for PA=0
                i1_lower[i] = i1_upper[i] = nnet->maxes[i]; //for PA=1
            } else {
                i0_lower[i] = i1_lower[i] = curr_subp.input.lower_matrix.data[i];
                i0_upper[i] = i1_upper[i] = curr_subp.input.upper_matrix.data[i];
                curr_volume *= (curr_subp.input.upper_matrix.data[i] - curr_subp.input.lower_matrix.data[i] + 1);
            }
        }

        struct Interval input0_interval = {
            (struct Matrix){i0_lower, 1, inputSize},
            (struct Matrix){i0_upper, 1, inputSize},
        };
        struct Interval input1_interval = {
            (struct Matrix){i1_lower, 1, inputSize},
            (struct Matrix){i1_upper, 1, inputSize},
        };

        float o0_lower[outputSize], o0_upper[outputSize];
        float o1_lower[outputSize], o1_upper[outputSize];

        struct Interval output0_interval = {
            (struct Matrix){o0_lower, outputSize, 1},
            (struct Matrix){o0_upper, outputSize, 1}
        };
        struct Interval output1_interval = {
            (struct Matrix){o1_lower, outputSize, 1},
            (struct Matrix){o1_upper, outputSize, 1}
        };

        //R to store neuron behavior
        int R0[numLayers][maxLayerSize];
        int R1[numLayers][maxLayerSize];

        //see if subp is fair or unfair
        int fairConc, unfairConc, fair0, fair1, unfair0, unfair1;
        fairConc = unfairConc = fair0 = fair1 = unfair0 = unfair1 = 0;

        // first check if this is a concrete data point (i.e. curr_vol = 1)
        if (curr_volume == 1) { // just do a concrete forward prop, it will be either fair or unfair            
            forward_prop(nnet, &input0_interval.lower_matrix, &output0_interval.lower_matrix);
            forward_prop(nnet, &input1_interval.lower_matrix, &output1_interval.lower_matrix);

            int out0Pos = (output0_interval.lower_matrix.data[0] > 0);
            int out1Pos = (output1_interval.lower_matrix.data[0] > 0);

            if (out0Pos == out1Pos) {
                fairConc = 1;
            }
            else {
                unfairConc = 1;
            }
        }
        
        // otherwise we do a normal symbolic forward prop
        else {
            memset(R0, 0, sizeof(float)*numLayers*maxLayerSize);
            memset(R1, 0, sizeof(float)*numLayers*maxLayerSize);

            // run forward pass for input0 and input1
            forward_prop_fair(nnet, &input0_interval, &output0_interval, R0);
            forward_prop_fair(nnet, &input1_interval, &output1_interval, R1);

            // if we have one output node to determine positive (above 0) or negative (below 0) label
            // note that the threshold is not 0.5, because we are looking at the input to the sigmoid function, hence 0 instead
            
            // fair if output0 and output 1 are both positive or negative
            fair0 = (output0_interval.lower_matrix.data[0] > 0 && output1_interval.lower_matrix.data[0] > 0);
            fair1 = (output0_interval.upper_matrix.data[0] < 0 && output1_interval.upper_matrix.data[0] < 0);

            // unfair if one of the outputs returns positive and the other returns negative
            unfair0 = (output0_interval.lower_matrix.data[0] > 0 && output1_interval.upper_matrix.data[0] < 0);
            unfair1 = (output1_interval.lower_matrix.data[0] > 0 && output0_interval.upper_matrix.data[0] < 0);
        }


        // with the outputs, calculate the rates

        // if fair
        if (fair0 || fair1 || fairConc){
            cert_volume += curr_volume;
            uncer_volume -= curr_volume;
        }

        // if unfair
        else if (unfair0 || unfair1 || unfairConc){
            fals_volume += curr_volume;
            uncer_volume -= curr_volume;
        }

        // if unknown (not determined to be fair or unfair)
        else {
            int shouldSplit = 1;    

            // 1. if we have reached MIN_CHECK_DEPTH, then we check for adv
            if (curr_subp.depth >= MIN_CHECK_DEPTH){
                int adv_found = check_adv(nnet, &curr_subp);

                // if this subp has adv, then we add to adv rate and dismiss this subp
                if (adv_found) {
                    shouldSplit = 0; // no more splitting!
                    num_adv += 1;
                    adv_volume += curr_volume;
                    uncer_volume -= curr_volume;
                } 
                // else, we should split further since we are still uncertain, hence no modification
            }
            
            // 2. if we have reached MAX_SPLIT_DEPTH, then we don't split
            if (curr_subp.depth >= MAX_SPLIT_DEPTH){
                shouldSplit = 0; // no more splitting!
            }
            

            // split if we shouldSplit
            if (shouldSplit) {

                // 1. calculate the grads based on backward_prop
                
                float g0_upper[inputSize], g0_lower[inputSize];
                float g1_upper[inputSize], g1_lower[inputSize];

                struct Interval grad0_interval = {
                    (struct Matrix){g0_lower, 1, inputSize},
                    (struct Matrix){g0_upper, 1, inputSize}
                };
                struct Interval grad1_interval = {
                    (struct Matrix){g1_lower, 1, inputSize},
                    (struct Matrix){g1_upper, 1, inputSize}
                };

                backward_prop(nnet, &grad0_interval, R0, 0);
                backward_prop(nnet, &grad1_interval, R1, 0);
              

                // 2. split feature based on grad-based smear value calculation
                struct Interval gradset[2] = {grad0_interval, grad1_interval};
                int sizeOfGradSet = 2;

                int split_feature = -1;
                float largest_smear_sum = 0;

                for (int i=0; i<feature_range_length; i++) { // for each feature
                    float interval_range = curr_subp.input.upper_matrix.data[feature_range[i]] - curr_subp.input.lower_matrix.data[feature_range[i]];
                    float smear_sum = 0;

                    for (int g=0; g<sizeOfGradSet; g++) { // for every grad
                        float e = (gradset[g].upper_matrix.data[feature_range[i]] > -gradset[g].lower_matrix.data[feature_range[i]])?\
                                    gradset[g].upper_matrix.data[feature_range[i]] : -gradset[g].lower_matrix.data[feature_range[i]]; // take the bigger coefficient (always positive)

                        float smear = e * interval_range;
                        smear_sum += smear;
                    }
                    
                    // update split_feature
                    if (largest_smear_sum < smear_sum) {
                        largest_smear_sum = smear_sum;
                        split_feature = i;
                    }
                }
                

                // 3. add our two subproblems to our stack
                int upper = (int) curr_subp.input.upper_matrix.data[feature_range[split_feature]];
                int lower = (int) curr_subp.input.lower_matrix.data[feature_range[split_feature]];
                int middle = (upper!=lower)? (upper + lower)/2 : upper;

                // save the curr_subp ranges into tmp
                float tmp_lower[inputSize], tmp_upper[inputSize];
                memcpy(tmp_lower, curr_subp.input.lower_matrix.data, sizeof(float)*inputSize);
                memcpy(tmp_upper, curr_subp.input.upper_matrix.data, sizeof(float)*inputSize);

                // initialize our next two subproblems
                memcpy(input_lowers[subproblem_remaining], tmp_lower, sizeof(float)*inputSize);
                memcpy(input_uppers[subproblem_remaining], tmp_upper, sizeof(float)*inputSize); // update this
                memcpy(input_lowers[subproblem_remaining+1], tmp_lower, sizeof(float)*inputSize); // and this to middle value for selected feature
                memcpy(input_uppers[subproblem_remaining+1], tmp_upper, sizeof(float)*inputSize);

                input_uppers[subproblem_remaining][feature_range[split_feature]] = (float) middle;
                input_lowers[subproblem_remaining+1][feature_range[split_feature]] = (float) (middle+1);
                
                // add first subp to our pq
                pq_intervals[subproblem_remaining] = (struct Interval) {
                    (struct Matrix){input_lowers[subproblem_remaining], 1, nnet->inputSize}, 
                    (struct Matrix){input_uppers[subproblem_remaining], 1, nnet->inputSize}
                };
                gettimeofday(&curr, NULL); time_spent = ((float)(curr.tv_sec - start.tv_sec) * 1000000 + (float)(curr.tv_usec - start.tv_usec)) / 1000000;                
                pq_subproblems[subproblem_remaining] = (struct Subproblem){pq_intervals[subproblem_remaining], curr_subp.depth+1, time_spent};


                // update counters
                subproblem_remaining++;
                subproblem_total++;

                //add second subp to our pq
                pq_intervals[subproblem_remaining] = (struct Interval) {
                    (struct Matrix){input_lowers[subproblem_remaining], 1, nnet->inputSize}, 
                    (struct Matrix){input_uppers[subproblem_remaining], 1, nnet->inputSize}
                };
                gettimeofday(&curr, NULL); time_spent = ((float)(curr.tv_sec - start.tv_sec) * 1000000 + (float)(curr.tv_usec - start.tv_usec)) / 1000000;            
                pq_subproblems[subproblem_remaining] = (struct Subproblem){pq_intervals[subproblem_remaining], curr_subp.depth+1, time_spent};

                // update counters
                subproblem_remaining++;
                subproblem_total++;
            }
        }
    }

    // done with verification, print and save results
    gettimeofday(&finish, NULL);
    time_spent = ((float)(finish.tv_sec - start.tv_sec) * 1000000 + (float)(finish.tv_usec - start.tv_usec)) / 1000000;
    rateCert = (double) cert_volume / (double) nnet->global_volume * 100;             // region for which model is def fair
    rateFals = (double) (fals_volume + num_adv) / (double) nnet->global_volume * 100; // region for which model is def unfair
    rateAdv = (double) (adv_volume - num_adv) / (double) nnet->global_volume * 100;   // region (minus the actual counterexamples) for which model outputs some counterexample - could be SAT/UNSAT
    rateUncer = (double) uncer_volume / (double) nnet->global_volume * 100;           // region for which we have not looked yet
    
    char buffer1[256];
    snprintf(buffer1, sizeof(buffer1), "%.2f sec | %d solved subps | %llu advs\n",
             time_spent, subproblem_total - subproblem_remaining, num_adv);

    char buffer2[256];
    snprintf(buffer2, sizeof(buffer2), "cer_rate: %.2f%%, fal_rate: %.2f%%, und_rate: %.2f%%\n",
             rateCert, rateFals, rateAdv + rateUncer);
    
    fprintf(stdout, "%s", buffer1);
    fprintf(stdout, "%s", buffer2);

    // create a file
    char *filename = strrchr(FULL_NET_PATH, '/') + 1; // e.g., points to 'm' in model.nnet
    char filenameWithoutExt[32]; // Assuming the filename without extension is short enough
    snprintf(filenameWithoutExt, sizeof(filenameWithoutExt), "%.*s", 
             (int)(strchr(filename, '.') - filename), filename);
    
    char filePath[64];
    snprintf(filePath, sizeof(filePath), "res/%s-%d.txt", 
             filenameWithoutExt, SENS_FEATURE_IDX);

    FILE *file = fopen(filePath, "w");
    if (file == NULL) {
        perror("Error creating the file");
        exit(1);
    }
    fprintf(file, "%s", buffer1);
    fprintf(file, "%s", buffer2);
    fclose(file);

    // free
    destroy_network(nnet);
    free(feature_range);
    free(pq_intervals);
    free(pq_subproblems);

    return 0;
}
