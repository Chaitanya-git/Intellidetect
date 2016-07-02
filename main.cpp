#include <iostream>
#include <armadillo>
#include "IntelliDetect.h"

int main(){
    cout<<"ANN Test Using the MNIST dataset"<<endl;
    network* net = new network();
    //net->load("/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_train.csv", 0, 9999);
    //net->train();
    net->predict("/home/chaitanya/Downloads/6.pgm").print();
    return 0;
}
