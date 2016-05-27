#ifndef INTELLIDETECT_H_INCLUDED
#define INTELLIDETECT_H_INCLUDED
#include <armadillo>
#include <cstring>

/* This header file contains definitions for functions for handling various ANN processes */

using namespace arma;
using namespace std;

class network{
        mat X,Y;
        mat Theta1, Theta2, nn_params;
        int inputLayerSize,HiddenLayerSize,OutputLayerSize;
        double lambda, alpha, mu;//-alpha*lambda; //0.995; //regularization parameter and learning rate and a momentum constant
        string inpPath;
    public:
        network(int, int, int, double, double, double);
        mat sigmoid(mat z);
        mat sigmoidGradient(mat z);
        mat randInitWeights(int Lin, int Lout);
        mat predict(mat Theta1, mat Theta2, mat Input);
        mat backpropogate(mat &nn_params,int InputLayerSize,int HiddenLayerSize,int OutputLayerSize,mat Inputs, mat Outputs, double lambda, double alpha);
        void train();
        void load(string path, int startInd=0, int endInd=0);
        double accuracy(mat Theta1, mat Theta2, mat X, mat Y);
};
network::network(int inputLayerSize = 784, int HiddenLayerSize = 100, int OutputLayerSize = 10,
                 double lambda = 1,double alpha = 2.5,double mu = 1){

    if(!nn_params.load("parameters2.csv")){
        Theta1 = randInitWeights(HiddenLayerSize, inputLayerSize+1);
        Theta2 = randInitWeights(OutputLayerSize,HiddenLayerSize+1);
        nn_params = join_vert(vectorise(Theta1),vectorise(Theta2)); //the weights in a more manageable format
    }
    else{
        cout<<"Loading Network sizes from file.";
//        inputLayerSize = as_scalar(nn_params(0));
//        HiddenLayerSize = as_scalar(nn_params(1));
//        OutputLayerSize = as_scalar(nn_params(2));
//        nn_params = nn_params.rows(3,nn_params.n_rows-1);
        Theta1 = reshape(nn_params.rows(0,(inputLayerSize+1)*(HiddenLayerSize)-1),HiddenLayerSize,inputLayerSize+1);
        Theta2 = reshape(nn_params.rows((inputLayerSize+1)*(HiddenLayerSize),nn_params.size()-1), OutputLayerSize, HiddenLayerSize+1);
    }
    cout<<"Weights Initialized"<<endl;
}

mat network::sigmoid(mat z){
    return 1.0/(1.0 + exp(-z));
}

mat network::sigmoidGradient(mat z){
    mat g = sigmoid(z);
    return g%(1-g);         // '%'is the overloaded element wise multiplication operator
}

mat network::randInitWeights(int Lin, int Lout){
    double epsilon = 0.52;
    return randu(Lin,Lout)*(2*epsilon) - epsilon;
}

mat network::predict(mat Theta1, mat Theta2, mat Input){
    int InputSize = Input.n_rows;
    Input = join_horiz(ones<mat>(InputSize,1),Input);
    mat h1 = join_horiz(ones<mat>(InputSize,1),sigmoid(Input*trans(Theta1)));
    mat h2 = sigmoid(h1*trans(Theta2));
    mat pred = zeros(InputSize,1);
    uword index;
    for(int i=0; i<InputSize; ++i){
        h2.row(i).max(index);
        pred(i) = index;
    }
    return pred;
}

mat network::backpropogate(mat &nn_params,int InputLayerSize,int HiddenLayerSize,int OutputLayerSize,mat Inputs, mat Outputs, double lambda, double alpha){
    mat Theta1 = reshape(nn_params.rows(0,(InputLayerSize+1)*(HiddenLayerSize)-1),HiddenLayerSize,InputLayerSize+1);
    mat Theta2 = reshape(nn_params.rows((InputLayerSize+1)*(HiddenLayerSize),nn_params.n_rows-1), OutputLayerSize, HiddenLayerSize+1);
    int InputSize = Inputs.n_rows;
    long double cost = 0;
    mat Theta1_grad = zeros<mat>(size(Theta1));
    mat Theta2_grad = zeros<mat>(size(Theta2));
    Inputs = join_horiz(ones<mat>(InputSize,1), Inputs); //Add the weights from the bias neuron.
    mat output_tmp = zeros<mat>(10,1);
    for(int i=0; i<InputSize; ++i){
        mat CurrentInput = trans(Inputs.row(i));
        mat z2 = Theta1*CurrentInput;
        mat a2 = sigmoid(z2);
        a2 = join_vert(ones<mat>(1,1),a2);
        mat z3 = Theta2*a2;
        mat a3 = sigmoid(z3);
        output_tmp(as_scalar(Outputs(i))) = 1;

        cost += as_scalar(accu(output_tmp%log(a3)+(1-output_tmp)%log(1-a3)))/InputSize*(-1);

        mat delta_3 = a3-output_tmp;
        mat delta_2 = trans(Theta2.cols(1,Theta2.n_cols-1))*delta_3%sigmoidGradient(z2);

        Theta1_grad += delta_2*CurrentInput.t();
        Theta2_grad += delta_3*a2.t();
        output_tmp(as_scalar(Outputs(i)),0) = 0;
    }
    cout<<"\tCost(unregularized) = "<<cost;
    cost += (accu(square(Theta1.cols(1,Theta1.n_cols-1)))+accu(square(Theta2.cols(1,HiddenLayerSize))))*lambda/(2*InputSize);
    cout<<"\t\tCost (regularized) = "<<cost<<endl;
    Theta1_grad /= InputSize;
    Theta2_grad /= InputSize;

    Theta1_grad += join_horiz(zeros<mat>(Theta1.n_rows,1), (lambda/InputSize)*Theta1.cols(1,Theta1.n_cols-1));
    Theta2_grad += join_horiz(zeros<mat>(Theta2.n_rows,1), (lambda/InputSize)*Theta2.cols(1,Theta2.n_cols-1));

    mat grad = join_vert(vectorise(Theta1_grad),vectorise(Theta2_grad));
    //nn_params -= alpha*grad;
    return grad;
}

void network::train(){
    int Total = X.n_rows, batch_size = X.n_rows/100;
    cout<<"\n\tStarting batch training.\n\n";

    cout<<"Prediction Accuracy before training: "<<accuracy(Theta1, Theta2, X, Y)<<endl<<endl;
    return;
    for(int k = 0;k<Total/batch_size; ++k){
        mat X_batch = X.rows(batch_size*(k),batch_size*(k+1)-1);
        mat Y_batch = Y.rows(batch_size*(k),batch_size*(k+1)-1);
        cout<<"Batch "<<k+1<<endl;
        for(int i=0; i<50; ++i){
            cout<<"\tIteration "<<i<<endl;
            nn_params = mu*nn_params - backpropogate(nn_params, inputLayerSize, HiddenLayerSize, OutputLayerSize, X_batch, Y_batch, lambda, alpha);
        }
    }
    Theta1 = reshape(nn_params.rows(0,(inputLayerSize+1)*(HiddenLayerSize)-1),HiddenLayerSize,inputLayerSize+1);
    Theta2 = reshape(nn_params.rows((inputLayerSize+1)*(HiddenLayerSize),nn_params.size()-1), OutputLayerSize, HiddenLayerSize+1);

    cout<<"Prediction Accuracy on training set: "<<accuracy(Theta1, Theta2, X, Y);
    cout<<"\n\nUsing test set"<<endl;
    load(inpPath);

    cout<<"Prediction Accuracy on test set: "<<accuracy(Theta1, Theta2, X, Y);
    mat hyper_params = {inputLayerSize,HiddenLayerSize,OutputLayerSize};
    nn_params = join_vert(vectorise(hyper_params),nn_params);
    nn_params.save("parameters4.csv",csv_ascii);
}

void network::load(string path, int startInd, int endInd){
    inpPath = path;
    X.load(path);
    X = shuffle(X);
    if(endInd)
        X = X.rows(startInd,endInd);
    else
        X = X.rows(startInd,X.n_rows-1);
    Y = X.col(0);
    X = X.cols(1,X.n_cols-1);
}

double network::accuracy(mat Theta1, mat Theta2, mat X, mat Y){
    umat prediction = (predict(Theta1, Theta2, X)==Y);
    return as_scalar(accu(prediction)*100.0/prediction.n_elem);
}

#endif // INTELLIDETECT_H_INCLUDED
