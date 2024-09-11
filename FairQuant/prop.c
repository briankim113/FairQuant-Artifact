#include <fenv.h>   //fesetround
#include <string.h> //memset, memcpy
#include "prop.h"

// following are helper functions from ReluDiff for forward prop
void affineTransform(struct Interval *interval, struct Matrix *posMatrix, struct Matrix *negMatrix,
                    struct Interval *outInterval, int overWrite) {

    if (overWrite) {
        /* source neuron --weight--> dest neuron */
        fesetround(FE_UPWARD); // compute upper bound
        /* For source neurons with a positive incoming weight, we multiply
         * by the source neuron's upper bound to maximize it. */
        matmul(          &interval->upper_matrix, posMatrix, &outInterval->upper_matrix);
        /* For source neurons with a negative incoming weight, we multiply
         * by the source neuron's lower bound to minimize it. */
        matmul_with_bias(&interval->lower_matrix, negMatrix, &outInterval->upper_matrix);

        fesetround(FE_DOWNWARD); // compute lower bound
        /* Use the lower bound when multiplying by a positive weight to minimize */
        matmul(          &interval->lower_matrix, posMatrix, &outInterval->lower_matrix);
        /* Use the upper bound when multiplying by negative weight to minimize */
        matmul_with_bias(&interval->upper_matrix, negMatrix, &outInterval->lower_matrix);
    } else {
        fesetround(FE_UPWARD);
        matmul_with_bias(&interval->upper_matrix, posMatrix, &outInterval->upper_matrix);
        matmul_with_bias(&interval->lower_matrix, negMatrix, &outInterval->upper_matrix);
        fesetround(FE_DOWNWARD);
        matmul_with_bias(&interval->lower_matrix, posMatrix, &outInterval->lower_matrix);
        matmul_with_bias(&interval->upper_matrix, negMatrix, &outInterval->lower_matrix);
    }
}

// from ReluDiff
void computeAllBounds(float *eqLow, float *eqUp, struct Interval *input, int inputSize,
                      float *low, float *lowsUp, float *upsLow, float *up) {

    float tempVal_lower = 0, tempVal_upper = 0, lower_s_upper = 0, upper_s_lower = 0;

    fesetround(FE_UPWARD); // Compute upper bounds
    for (int k = 0; k < inputSize; k++) {
        /* lower's upper bound */
        if (eqLow[k] >= 0) {
            /* If coefficient is positive, multiply it by the upper bound
             * of the input. */
            lower_s_upper += eqLow[k] * input->upper_matrix.data[k];
        } else {
            /* Otherwise, multiply by lower bound of the input. */
            lower_s_upper += eqLow[k] * input->lower_matrix.data[k];
        }
        // printf("%f, ", lower_s_upper);
        /* upper bound */
        if (eqUp[k] >= 0) {
            tempVal_upper += eqUp[k] * input->upper_matrix.data[k];
        } else {
            tempVal_upper += eqUp[k] * input->lower_matrix.data[k];
        }
        // printf("%f\n", tempVal_upper);
    }
    lower_s_upper += eqLow[inputSize];
    tempVal_upper += eqUp[inputSize];
    // printf("%f, ", lower_s_upper);
    // printf("%f\n", tempVal_upper);


    fesetround(FE_DOWNWARD); // Compute lower bounds
    for (int k = 0; k < inputSize; k++) {
        /* lower bound */
        if (eqLow[k] >= 0) {
            /* If coefficient is positive, multiply by lower bound
             * of the input. */
            tempVal_lower += eqLow[k] * input->lower_matrix.data[k];
        } else {
            /* Otherwise, multiply by upper bound. */
            tempVal_lower += eqLow[k] * input->upper_matrix.data[k];
        }
        // printf("%f, ", tempVal_lower);

        /* upper's lower bound */
        if(eqUp[k] >= 0) {
            upper_s_lower += eqUp[k] * input->lower_matrix.data[k];
        } else {
            upper_s_lower += eqUp[k] * input->upper_matrix.data[k];
        }
        // printf("%f\n", upper_s_lower);

    }
    tempVal_lower += eqLow[inputSize];
    upper_s_lower += eqUp[inputSize];
    // printf("%f, ", tempVal_lower);
    // printf("%f\n", upper_s_lower);

    *low = tempVal_lower;
    *lowsUp = lower_s_upper;
    *upsLow = upper_s_lower;
    *up = tempVal_upper;
}

// from ReluDiff
void zero_interval(struct Interval *interval, int eqSize, int neuron) {
    for (int k = 0; k < eqSize; k++) {
        interval->lower_matrix.data[k + neuron*(eqSize)] = 0;
        interval->upper_matrix.data[k + neuron*(eqSize)] = 0;
    }
}

/*
 * Concrete forward propagation with openblas
 * It takes in network and concrete input matrix.
 * Outputs the concrete outputs.
 */
int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output)
{
    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];
    struct Matrix Z = {z, 1, inputSize};
    struct Matrix A = {a, 1, inputSize};

    memcpy(Z.data, input->data, nnet->inputSize*sizeof(float));

    for(int layer=0;layer<numLayers;layer++){
        A.row = nnet->bias[layer].row;
        A.col = nnet->bias[layer].col;
        memcpy(A.data, nnet->bias[layer].data, A.row*A.col*sizeof(float));

        matmul_with_bias(&Z, &nnet->weights[layer], &A);
        if(layer<numLayers-1){
            relu(&A);
        }
        memcpy(Z.data, A.data, A.row*A.col*sizeof(float));
        Z.row = A.row;
        Z.col = A.col;
    }

    memcpy(output->data, A.data, A.row*A.col*sizeof(float));
    output->row = A.row;
    output->col = A.col;

    return 1;
}

/*
 * Backward propagation to calculate the gradient ranges of inputs.
 * Takes in network and gradient masks.
 * Outputs input gradient ranges.
 */
void backward_prop(struct NNet *nnet, struct Interval *grad, int R[][nnet->maxLayerSize], int y)
{
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

    float grad_upper[maxLayerSize];
    float grad_lower[maxLayerSize];
    float new_grad_upper[maxLayerSize];
    float new_grad_lower[maxLayerSize];

    memcpy(grad_upper, nnet->matrix[numLayers-1][0][y], sizeof(float)*maxLayerSize);
    memcpy(grad_lower, nnet->matrix[numLayers-1][0][y], sizeof(float)*maxLayerSize);

    for (int layer = numLayers-2; layer > -1; layer--){ 

        // for (int i=0; i<maxLayerSize; i++){
        //     printf("%f ", grad_lower[i]);
        // } printf("\n");
        // for (int i=0; i<maxLayerSize; i++){
        //     printf("%f ", grad_upper[i]);
        // } printf("\n");

        float **weights = nnet->matrix[layer][0];
        memset(new_grad_upper, 0, sizeof(float)*maxLayerSize);
        memset(new_grad_lower, 0, sizeof(float)*maxLayerSize);

        /* For each neuron in the _current_ layer
        * (layerSizes includes the input layer, hence the layer + 1).
        * In the nnet->weights matrix, layer is the _current_ layer.
        * In the nnet->layerSizes, layer is the _previous_ layer. */

        for (int j = 0; j < nnet->layerSizes[layer+1]; j++) {

            /* Perform ReLU */
            if(R[layer][j] == 0){
                //inactive, zero
                grad_upper[j] = grad_lower[j] = 0;
            }
            else if (R[layer][j] == 1) {
                //nonlinear, 0 or keep as is
                grad_upper[j] = (grad_upper[j]>0)?grad_upper[j]:0;
                grad_lower[j] = (grad_lower[j]<0)?grad_lower[j]:0;
            }
            else {
                //active, keep as is
            }

            /* Perform matrix multiplication */

            /* For each neuron in the _previous_ layer */
            for (int i = 0; i < nnet->layerSizes[layer]; i++) {
                if (weights[j][i] >= 0) {
                    /* Weight is positive
                        * Lower to lower, upper to upper */
                    new_grad_upper[i] += weights[j][i]*grad_upper[j]; 
                    new_grad_lower[i] += weights[j][i]*grad_lower[j]; 
                } else {
                    /* Else flip */
                    new_grad_upper[i] += weights[j][i]*grad_lower[j]; 
                    new_grad_lower[i] += weights[j][i]*grad_upper[j]; 
                }
            }
        }

        if (layer != 0) {
            memcpy(grad_upper, new_grad_upper, sizeof(float)*maxLayerSize);
            memcpy(grad_lower, new_grad_lower, sizeof(float)*maxLayerSize);
        }
        else {
            memcpy(grad->lower_matrix.data, new_grad_lower, sizeof(float)*inputSize);
            memcpy(grad->upper_matrix.data, new_grad_upper, sizeof(float)*inputSize);
        }

    }

    // printMatrix(&grad->lower_matrix);
    // printMatrix(&grad->upper_matrix);
}

/*
forward prop for individual fairness
*/
void forward_prop_fair(struct NNet *nnet, struct Interval *input, struct Interval *output, int (*R)[nnet->maxLayerSize]){
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    int maxLayerSize = nnet->maxLayerSize;

    int equationSize = inputSize + 1; // number of variables in equation
    int maxEquationMatrixSize = equationSize * maxLayerSize * sizeof(float); // number of bytes needed to store matrix of equations

    float *eqLow = malloc(maxEquationMatrixSize);
    float *eqUp = malloc(maxEquationMatrixSize);
    float *newEqLow = malloc(maxEquationMatrixSize);
    float *newEqUp = malloc(maxEquationMatrixSize);

    memset(eqUp, 0, maxEquationMatrixSize);
    memset(eqLow, 0, maxEquationMatrixSize);
    memset(newEqUp, 0, maxEquationMatrixSize);
    memset(newEqLow, 0, maxEquationMatrixSize);

    //initially inputSize, will become maxLayerSize down the layer
    struct Interval eqInterval = {
        (struct Matrix){eqLow, equationSize, inputSize},
        (struct Matrix){eqUp, equationSize, inputSize}
    };
    struct Interval newEqInterval = {
        (struct Matrix){newEqLow, equationSize, maxLayerSize},
        (struct Matrix){newEqUp, equationSize, maxLayerSize}
    };

    float concLow=0.0, concUp=0.0;
    float concUpsLow=0.0, concLowsUp=0.0;

    for (int i=0; i<inputSize; i++)
    {
        eqLow[i*equationSize+i] = 1;
        eqUp[i*equationSize+i] = 1;
    }

    for (int layer = 0; layer<(numLayers); layer++)
    {
        memset(newEqLow, 0, maxEquationMatrixSize);
        memset(newEqUp, 0, maxEquationMatrixSize);
        
        struct Matrix weights = nnet->weights[layer];
        struct Matrix bias = nnet->bias[layer];

        struct Matrix pWeights = nnet->posWeights[layer];
        struct Matrix nWeights = nnet->negWeights[layer];

        affineTransform(&eqInterval, &pWeights, &nWeights, &newEqInterval, 1);

        // +1 because this array includes input size
        for (int neuron=0; neuron < nnet->layerSizes[layer+1]; neuron++)
        {
            int eqOffset = neuron*equationSize;
            int constantIndex = eqOffset + equationSize - 1;

            concUp = concLow = 0.0;
            concLowsUp = concUpsLow = 0.0;
            
            /* Add bias to the constant */
            fesetround(FE_DOWNWARD); // lower bounds
            newEqLow[constantIndex] += bias.data[neuron];

            fesetround(FE_UPWARD); // upper bounds
            newEqUp[constantIndex] += bias.data[neuron];

            computeAllBounds(newEqLow + eqOffset, newEqUp + eqOffset, input, inputSize, &concLow, &concLowsUp, &concUpsLow, &concUp);

            //Perform ReLU
            if (layer < (numLayers-1)){
                if (concUp <= 0.0){ //inactive, zeroed out
                    R[layer][neuron] = 0;
                    zero_interval(&newEqInterval, equationSize, neuron);
                }
                else if(concLow >= 0.0){ //active, keep as is
                    R[layer][neuron] = 2;
                }
                else { //nonlinear
                    R[layer][neuron] = 1;

                    if (concUpsLow < 0.0){
                        //linear approximation of upper bound: ReLU <= (u / u-l)*(Eq-l)
                        double coeff = concUp / (concUp - concUpsLow);
                        for (int i=0; i<equationSize; i++){
                            newEqUp[eqOffset+i] *= coeff;
                        }
                        newEqUp[eqOffset+inputSize] -= coeff*concUpsLow;
                    }
                    else {
                        //keep upper bound as is
                    }

                    if (concLowsUp < 0.0){
                        //lower bound is 0
                        for (int i=0; i<equationSize; i++){
                            newEqLow[eqOffset+i] = 0;
                        }
                    }
                    else {
                        //linear approximation of lower bound: ReLU >= (u / u-l)*(Eq)
                        double coeff = concLowsUp / (concLowsUp - concLow);
                        for (int i=0;i<equationSize;i++){
                            newEqLow[eqOffset+i] *= coeff;
                        }
                    }
                }
            }
            else {
                output->upper_matrix.data[neuron] = concUp;
                output->lower_matrix.data[neuron] = concLow;
            }
        }

        memcpy(eqLow, newEqLow, maxEquationMatrixSize);
        memcpy(eqUp, newEqUp, maxEquationMatrixSize);
        eqInterval.lower_matrix.row = eqInterval.upper_matrix.row =\
                                                        newEqInterval.lower_matrix.row;
        eqInterval.lower_matrix.col = eqInterval.upper_matrix.col =\
                                                        newEqInterval.lower_matrix.col;
    }

    free(eqLow);
    free(eqUp);
    free(newEqLow);
    free(newEqUp);
}
