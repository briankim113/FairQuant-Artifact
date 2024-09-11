# FairQuant

## Installation

To run the tool, please refer to the [installation file](./INSTALL.md) for detailed instructions.


## Run the tool

Our tool can be easily run by following the steps listed below.

1. Run the following command `make all` to generate an executable file called `network_test`
2. Run the desired verification on any of the following datasets and their protected attribute:
   1. `./adult.sh ["sex"]`
   2. `./bank.sh ["age"]`
   3. `./german.sh ["age" (in paper) or "sex"]`
   4. `./compas.sh ["race" (in paper) or "sex" or "age"]`


## Sources

This source code for our tool FairQuant is adapted from the following artifacts:

1. [ReluVal (USENIX Security '18)](https://github.com/tcwangshiqi-columbia/ReluVal)
2. [ReluDiff (ICSE '20)](https://github.com/pauls658/ReluDiff-ICSE2020-Artifact)