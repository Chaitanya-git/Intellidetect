#ifndef INTELLIDETECT_H_INCLUDED
#define INTELLIDETECT_H_INCLUDED
#include <armadillo>

/* This header file contains definitions for functions for handling various ANN processes */

using namespace arma;

mat sigmoid(mat z){
    return 1.0/(1.0 + exp(-z));
}

mat sigmoidGradient(mat z){
    mat g = sigmoid(z);
    return g%(1-g);         // '%'is the overloaded element wise multiplication operator
}

mat randInitWeights(int Lin, int Lout){
    double epsilon = 0.52;
    return randu(Lin,Lout)*(2*epsilon) - epsilon;
}

mat predict(mat Theta1, mat Theta2, mat Input){
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

mat backpropogate(mat &nn_params,int InputLayerSize,int HiddenLayerSize,int OutputLayerSize,mat Inputs, mat Outputs, double lambda, double alpha){
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

#endif // INTELLIDETECT_H_INCLUDED
