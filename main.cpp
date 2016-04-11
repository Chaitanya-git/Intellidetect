#include <iostream>
#include <armadillo>
#include "IntelliDetect.h"

int main(){
    cout<<"ANN Test Using the MNIST dataset"<<endl;
    mat X,Y;
    mat Theta1,Theta2;
    X.load("/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_train.csv");
    Y = X.col(0);
    X = X.cols(1,X.n_cols-1);
    int inputLayerSize = 784;
    int HiddenLayerSize = 25;
    int OutputLayerSize = 10;
    Theta1 = randInitWeights(HiddenLayerSize, inputLayerSize+1);
    Theta2 = randInitWeights(OutputLayerSize,HiddenLayerSize+1);

    cout<<"Weights Initialized"<<endl;
    mat nn_parms = join_vert(vectorise(Theta1),vectorise(Theta2));
    //X = reshape(X.rows(0,X.n_rows-1),X.size()/784,784);
    cout<<X.n_cols<<endl<<X.n_rows<<endl;
    //Y= vectorise(Y);
    double lambda = 0.5, alpha = 0.01; //regularization parameter
    //mat grad =
    for(int i=0; i<50; ++i)
        backpropogate(nn_parms, inputLayerSize, HiddenLayerSize, OutputLayerSize, X, Y, lambda, alpha);
    //cout<<trans(Y);
    //cout<<endl<<endl<<X.rows(0,10);
    cout<<"\n\nUsing test set"<<endl;
    X.load("/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_test.csv");
    Y = X.col(0);
    X = X.cols(1,X.n_cols-1);
    backpropogate(nn_parms, inputLayerSize, HiddenLayerSize, OutputLayerSize, X, Y, lambda, alpha);
    return 0;
}
