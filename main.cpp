#include <iostream>
#include <armadillo>
#include "IntelliDetect.h"

int main(){
    cout<<"ANN Test Using the MNIST dataset"<<endl;
    mat X,Y;
    mat Theta1, Theta2, nn_params;
    X.load("/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_train.csv");
    X = shuffle(X);
    Y = X.col(0);
    X = X.cols(1,X.n_cols-1);
    int inputLayerSize = 784;
    int HiddenLayerSize = 49;
    int OutputLayerSize = 10;
    Theta1 = randInitWeights(HiddenLayerSize, inputLayerSize+1);
    Theta2 = randInitWeights(OutputLayerSize,HiddenLayerSize+1);

    cout<<"Weights Initialized"<<endl;
    if(!nn_params.load("parameters.csv"))
        nn_params = join_vert(vectorise(Theta1),vectorise(Theta2)); //the weights in a more manageable format
    //cout<<X.n_cols<<endl<<X.n_rows<<endl;

    double lambda = 1, alpha = 0.01; //regularization parameter and learning rate
    int Total = X.n_rows, batch_size = X.n_rows/100;
    cout<<"\n\tStarting batch training.\n\n";
    umat prediction = (predict(Theta1, Theta2, X)==Y);
    double accuracy = as_scalar(accu(prediction)*100.0/prediction.n_elem);
    cout<<"Prediction Accuracy before training: "<<accuracy<<endl<<endl;
    for(int k = 0;k<(Total/batch_size); ++k){
        mat X_batch = X.rows(batch_size*(k),batch_size*(k+1)-1);
        mat Y_batch = Y.rows(batch_size*(k),batch_size*(k+1)-1);
        cout<<"Batch "<<k+1<<endl;
        for(int i=0; i<10; ++i){
            cout<<"\tIteration "<<i<<endl;
            nn_params -= backpropogate(nn_params, inputLayerSize, HiddenLayerSize, OutputLayerSize, X_batch, Y_batch, lambda, alpha);
        }
    }
    Theta1 = reshape(nn_params.rows(0,(inputLayerSize+1)*(HiddenLayerSize)-1),HiddenLayerSize,inputLayerSize+1);
    Theta2 = reshape(nn_params.rows((inputLayerSize+1)*(HiddenLayerSize),nn_params.size()-1), OutputLayerSize, HiddenLayerSize+1);

    prediction = (predict(Theta1, Theta2, X)==Y);
    accuracy = as_scalar(accu(prediction)*100.0/prediction.n_elem);
    cout<<"Prediction Accuracy on training set: "<<accuracy;
    cout<<"\n\nUsing test set"<<endl;
    X.load("/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_test.csv");
    Y = X.col(0);
    X = X.cols(1,X.n_cols-1);

    prediction = (predict(Theta1, Theta2, X)==Y);
    accuracy = as_scalar(accu(prediction)*100.0/prediction.n_elem);
    cout<<"Prediction Accuracy on test set: "<<accuracy;

    nn_params.save("parameters.csv",csv_ascii);
    return 0;
}
