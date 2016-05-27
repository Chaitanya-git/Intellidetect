#include <iostream>
#include <armadillo>
#include "IntelliDetect.h"

int main(){
    cout<<"ANN Test Using the MNIST dataset"<<endl;
    mat X,Y;
    mat Theta1, Theta2, nn_params;
    load(X, Y, "/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_train.csv", 0, 9999);
    int inputLayerSize = 784;
    int HiddenLayerSize = 100;
    int OutputLayerSize = 10;
    double lambda = 0.1, alpha = 2.5, mu = 1-alpha*lambda; //0.995; //regularization parameter and learning rate and a momentum constant
    cout<<"Weights Initialized"<<endl;
//    if(!nn_params.load("parameters.csv")){
//        Theta1 = randInitWeights(HiddenLayerSize, inputLayerSize+1);
//        Theta2 = randInitWeights(OutputLayerSize,HiddenLayerSize+1);
//        nn_params = join_vert(vectorise(Theta1),vectorise(Theta2)); //the weights in a more manageable format
//    }
//    else{
//        cout<<"Loading Network sizes from file.";
//        inputLayerSize = as_scalar(nn_params(0));
//        HiddenLayerSize = as_scalar(nn_params(1));
//        OutputLayerSize = as_scalar(nn_params(2));
//        nn_params = nn_params.rows(3,nn_params.n_rows-1);
//        Theta1 = reshape(nn_params.rows(0,(inputLayerSize+1)*(HiddenLayerSize)-1),HiddenLayerSize,inputLayerSize+1);
//        Theta2 = reshape(nn_params.rows((inputLayerSize+1)*(HiddenLayerSize),nn_params.size()-1), OutputLayerSize, HiddenLayerSize+1);
//    }
    int Total = X.n_rows, batch_size = X.n_rows/10;
    cout<<"\n\tStarting batch training.\n\n";
    double prev_acc=0,opt_alpha=0;

    cout<<"finding best learning rate."<<endl;
    for(alpha = 0.01; alpha<3; alpha +=0.01){
        Theta1 = randInitWeights(HiddenLayerSize, inputLayerSize+1);
        Theta2 = randInitWeights(OutputLayerSize,HiddenLayerSize+1);
        nn_params = join_vert(vectorise(Theta1),vectorise(Theta2)); //the weights in a more manageable format
        cout<<"Prediction Accuracy before training: "<<accuracy(Theta1, Theta2, X, Y)<<endl<<endl;
        for(int k = 0;k<Total/batch_size; ++k){
            mat X_batch = X.rows(batch_size*(k),batch_size*(k+1)-1);
            mat Y_batch = Y.rows(batch_size*(k),batch_size*(k+1)-1);
            cout<<"Batch "<<k+1<<endl;
            for(int i=0; i<10; ++i){
                cout<<"\tIteration "<<i<<endl;
                nn_params = mu*nn_params - backpropogate(nn_params, inputLayerSize, HiddenLayerSize, OutputLayerSize, X_batch, Y_batch, lambda, alpha);
            }
        }
        Theta1 = reshape(nn_params.rows(0,(inputLayerSize+1)*(HiddenLayerSize)-1),HiddenLayerSize,inputLayerSize+1);
        Theta2 = reshape(nn_params.rows((inputLayerSize+1)*(HiddenLayerSize),nn_params.size()-1), OutputLayerSize, HiddenLayerSize+1);
        double acc = accuracy(Theta1, Theta2, X, Y);
        cout<<"Prediction Accuracy on training set: "<<acc;
        if(acc>prev_acc){
            prev_acc = acc;
            opt_alpha = alpha;
        }
    }
    alpha = opt_alpha;
    cout<<"finding best regularization parameter."<<endl;
    load(X, Y, "/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_train.csv", 10000, 19999);
    double opt_lambda = 0;
    prev_acc = 0;
    for(lambda = 0.01; lambda<1; lambda +=0.01){
        Theta1 = randInitWeights(HiddenLayerSize, inputLayerSize+1);
        Theta2 = randInitWeights(OutputLayerSize,HiddenLayerSize+1);
        nn_params = join_vert(vectorise(Theta1),vectorise(Theta2)); //the weights in a more manageable format
        cout<<"Prediction Accuracy before training: "<<accuracy(Theta1, Theta2, X, Y)<<endl<<endl;
        for(int k = 0;k<Total/batch_size; ++k){
            mat X_batch = X.rows(batch_size*(k),batch_size*(k+1)-1);
            mat Y_batch = Y.rows(batch_size*(k),batch_size*(k+1)-1);
            cout<<"Batch "<<k+1<<endl;
            for(int i=0; i<10; ++i){
                cout<<"\tIteration "<<i<<endl;
                nn_params = mu*nn_params - backpropogate(nn_params, inputLayerSize, HiddenLayerSize, OutputLayerSize, X_batch, Y_batch, lambda, alpha);
            }
        }
        Theta1 = reshape(nn_params.rows(0,(inputLayerSize+1)*(HiddenLayerSize)-1),HiddenLayerSize,inputLayerSize+1);
        Theta2 = reshape(nn_params.rows((inputLayerSize+1)*(HiddenLayerSize),nn_params.size()-1), OutputLayerSize, HiddenLayerSize+1);
        double acc = accuracy(Theta1, Theta2, X, Y);
        cout<<"Prediction Accuracy on training set: "<<acc;
        if(acc>prev_acc){
            prev_acc = acc;
            opt_lambda = lambda;
        }
    }
    lambda = opt_lambda;
    load(X, Y, "/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_train.csv",0,9999);
    Theta1 = randInitWeights(HiddenLayerSize, inputLayerSize+1);
    Theta2 = randInitWeights(OutputLayerSize,HiddenLayerSize+1);
    nn_params = join_vert(vectorise(Theta1),vectorise(Theta2)); //the weights in a more manageable format
    cout<<"Prediction Accuracy before training: "<<accuracy(Theta1, Theta2, X, Y)<<endl<<endl;
    for(int k = 0;k<Total/batch_size; ++k){
        mat X_batch = X.rows(batch_size*(k),batch_size*(k+1)-1);
        mat Y_batch = Y.rows(batch_size*(k),batch_size*(k+1)-1);
        cout<<"Batch "<<k+1<<endl;
        for(int i=0; i<10; ++i){
            cout<<"\tIteration "<<i<<endl;
            nn_params = mu*nn_params - backpropogate(nn_params, inputLayerSize, HiddenLayerSize, OutputLayerSize, X_batch, Y_batch, lambda, alpha);
        }
    }
    Theta1 = reshape(nn_params.rows(0,(inputLayerSize+1)*(HiddenLayerSize)-1),HiddenLayerSize,inputLayerSize+1);
    Theta2 = reshape(nn_params.rows((inputLayerSize+1)*(HiddenLayerSize),nn_params.size()-1), OutputLayerSize, HiddenLayerSize+1);
    double acc = accuracy(Theta1, Theta2, X, Y);
    cout<<"Prediction Accuracy on training set: "<<acc;

    cout<<"\n\nUsing test set"<<endl;
    load(X, Y, "/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_test.csv");

    cout<<"Prediction Accuracy on test set: "<<accuracy(Theta1, Theta2, X, Y);
    mat hyper_params = {inputLayerSize,HiddenLayerSize,OutputLayerSize};
    nn_params = join_vert(vectorise(hyper_params),nn_params);
    nn_params.save("parameters.csv",csv_ascii);

    cout<<"alpha = "<<alpha;
    cout<<"lambda = "<<lambda;
    return 0;
}
