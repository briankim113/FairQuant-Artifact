`models` contains the .nnet and .h5 files used to represent DNNs to verify (for FairQuant and Fairify, respectively).

- `adult`, `bank`, and `german` contain the TensorFlow networks, originally provided by Fairify in .h5 files which we turn into .nnet files. 
- `compas` contain our newly trained TensorFlow networks in .h5 files which we turn into .nnet files.
    - `compas-7` files are compressed as .tar.gz files, so please extract them by running the following command:
    ```shell
    cd compas
    tar -xzf compas-7.h5.tar.gz 
    tar -xzf compas-7.nnet.tar.gz
    ```
- `example.nnet` is the toy network presented in the paper.
