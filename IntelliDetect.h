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
    double epsilon = 0.12;
    return randg(Lin,Lout)*(2*epsilon) - epsilon;
}

mat predict(mat Theta1, mat Theta2, mat Input){
    int InputSize = Input.n_rows;
    Input = join_horiz(ones<mat>(InputSize,1),Input);
    mat h1 = join_horiz(ones<mat>(InputSize,1),Input*trans(Theta1));
    mat h2 = h1*trans(Theta2);
    mat pred = zeros(InputSize,1);
    uword index;
    for(int i=0; i<InputSize; ++i){
        h2.row(i).max(index);
        pred(i) = index;
    }
    return pred;
}

void backpropogate(mat &nn_params,int InputLayerSize,int HiddenLayerSize,int OutputLayerSize,mat Inputs, mat Outputs, double lambda, double alpha){
    mat Theta1 = reshape(nn_params.rows(0,InputLayerSize*(HiddenLayerSize)-1),HiddenLayerSize,InputLayerSize+1);
    mat Theta2 = reshape(nn_params.rows(InputLayerSize*(HiddenLayerSize),nn_params.size()-1), OutputLayerSize, HiddenLayerSize+1);
    int InputSize = Inputs.n_rows;
    long double cost = 0;
    mat Theta1_grad = zeros<mat>(size(Theta1));
    mat Theta2_grad = zeros<mat>(size(Theta2));
    Inputs = join_horiz(ones<mat>(InputSize,1), Inputs); //Add the weights from the bias neuron.
    cout<<"\t Entering loop!"<<endl;
    for(int i=0; i<InputSize; ++i){
        mat CurrentInput = trans(Inputs.row(i));
        mat z2 = Theta1*CurrentInput;
        //cout<<"\tz2 computed."<<endl;
        mat a2 = sigmoid(z2);
        a2 = join_vert(ones<mat>(1,1),a2);
        mat z3 = Theta2*a2;
        //cout<<"\tz3 computed."<<endl;
        mat a3 = sigmoid(z3);
        //mat pred = zeros<mat>(OutputLayerSize,1);
//        uword index;
//        for(int j=0; j<OutputLayerSize; ++j){
//            a3.row(j).max(index);
//            pred(j) = index;
//        }
        mat output_tmp = zeros<mat>(10,1);
        output_tmp(as_scalar(Outputs(i)),0) = 1;
        //cout<<"Pred computed"<<endl;
        //cout<<"Output_tmp = "<<output_tmp<<endl;
        cost += as_scalar(accu(output_tmp%log(a3)+(1-output_tmp)%log(1-a3)))/InputSize*-1;

        //cout<<"Cost Updated to "<<cost<<endl;


        mat delta_3 = a3-output_tmp;
        //cout<<"Size of delta_3 = "<<size(delta_3)<<endl
          //  <<"Size of sigGrad(z2) = "<<size(sigmoidGradient(z2));
        mat delta_2 = trans(Theta2.cols(1,HiddenLayerSize))*delta_3%sigmoidGradient(z2);
        //cout<<"\tdelta_2 computed."<<endl;

        Theta1_grad += delta_2*CurrentInput.t();
        Theta2_grad += delta_3*a2.t();
        //cout<<"\tgrads computed."<<endl;
    }
    cout<<"Cost(unregularized) = "<<cost<<endl;
    cost += (accu(Theta1.cols(1,InputLayerSize))+accu(Theta2.cols(1,HiddenLayerSize)))*lambda/(2*InputSize);
    cout<<"Cost (regularized) = "<<cost<<endl;
    Theta1_grad /= InputSize;
    Theta2_grad /= InputSize;

    Theta1_grad += join_horiz(zeros<mat>(Theta1.n_rows,1), (lambda/InputSize)*Theta1.cols(1,InputLayerSize));
    Theta2_grad += join_horiz(zeros<mat>(Theta2.n_rows,1), (lambda/InputSize)*Theta2.cols(1,HiddenLayerSize));

    mat grad = join_vert(vectorise(Theta1_grad),vectorise(Theta2_grad));
    nn_params -= alpha*grad;
    //return grad;
}

#endif // INTELLIDETECT_H_INCLUDED
