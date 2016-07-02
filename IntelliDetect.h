#ifndef INTELLIDETECT_H_INCLUDED
#define INTELLIDETECT_H_INCLUDED
#include <armadillo>
#include <cstring>

/* This header file contains definitions for functions for handling various ANN processes */

using namespace arma;
using namespace std;

class network{
        mat m_X,m_Y;
        mat m_Theta1, m_Theta2, m_nn_params;
        int m_inputLayerSize,m_hiddenLayerSize,m_outputLayerSize;
        double m_lambda, m_alpha, m_mu;//-alpha*lambda; //0.995; //regularization parameter and learning rate and a momentum constant
        string inpPath;
    public:
        network(int, int, int, double, double, double);
        mat sigmoid(mat z);
        mat sigmoidGradient(mat z);
        mat randInitWeights(int Lin, int Lout);
        mat predict(mat Input);
        mat predict(string input);
        mat output(string input);
        mat output(mat);
        mat backpropogate(mat Inputs, mat Outputs);
        void train();
        void load(string path, int startInd=0, int endInd=0);
        double accuracy(mat &m_X, mat &m_Y);
};
network::network(int inpSize = 784, int HdSize = 100, int OpSize = 10,
                 double lm = 1,double al = 2.5,double m = 1){
    m_inputLayerSize = inpSize;
    m_hiddenLayerSize = HdSize;
    m_outputLayerSize = OpSize;
    m_lambda = lm;
    m_alpha = al;
    m_mu = m;
    if(m_nn_params.load("parameters2.csv")==false){
        cout<<"Randomly initialising weights."<<endl;
        m_Theta1 = randInitWeights(m_hiddenLayerSize, m_inputLayerSize+1);
        m_Theta2 = randInitWeights(m_outputLayerSize,m_hiddenLayerSize+1);
        m_nn_params = join_vert(vectorise(m_Theta1),vectorise(m_Theta2)); //the weights in a more manageable format
    }
    else{
        cout<<"Loading Network sizes from file."<<endl;
//        inputLayerSize = as_scalar(nn_params(0));
//        hiddenLayerSize = as_scalar(nn_params(1));
//        outputLayerSize = as_scalar(nn_params(2));
//        nn_params = nn_params.rows(3,nn_params.n_rows-1);
        m_Theta1 = reshape(m_nn_params.rows(0,(m_inputLayerSize+1)*(m_hiddenLayerSize)-1),m_hiddenLayerSize,m_inputLayerSize+1);
        m_Theta2 = reshape(m_nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSize),m_nn_params.size()-1), m_outputLayerSize, m_hiddenLayerSize+1);
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

mat network::predict(mat Input){
    int InputSize = Input.n_rows;
    Input = join_horiz(ones<mat>(InputSize,1),Input);
    mat z2 = Input*trans(m_Theta1);
    mat h1 = join_horiz(ones<mat>(z2.n_rows,1),sigmoid(z2));
    mat h2 = sigmoid(h1*trans(m_Theta2));
    mat pred = zeros(InputSize,1);
    uword index;
    for(int i=0; i<InputSize; ++i){
        h2.row(i).max(index);
        pred(i) = index;
    }
    return pred;
}
mat network::output(mat Input){
    int InputSize = Input.n_rows;
    Input = join_horiz(ones<mat>(InputSize,1),Input);
    mat z2 = Input*trans(m_Theta1);
    mat h1 = join_horiz(ones<mat>(z2.n_rows,1),sigmoid(z2));
    mat h2 = sigmoid(h1*trans(m_Theta2));
    mat pred = zeros(InputSize,1);
    cout<<"Output: "<<h2;
    for(int i=0; i<InputSize; ++i){
        pred(i) = h2.row(i).max();
    }
    return pred;

}
mat network::predict(string input){
    mat inputMat;
    inputMat.load(input);
    //inputMat = X.row(200);
    //inputMat = inputMat.t();
    cout<<inputMat.size()<<endl;
    umat tmpMat = conv_to<umat>::from(inputMat);
    tmpMat.reshape(28,28);
    for(int i=0;i<28;++i){
        for(int j=0;j<28;++j)
            cout<<as_scalar(tmpMat.at(i,j))<<" ";
        cout<<endl;
    }
    //cout<<endl<<"Y=: "<<as_scalar(Y.row(200))<<endl;
    inputMat.reshape(1,784);
    return predict(inputMat);
}

mat network::output(string input){
    mat inputMat;
    inputMat.load(input);
    //inputMat = X.row(200);
    //inputMat = inputMat.t();
    cout<<inputMat.size()<<endl;
    umat tmpMat = conv_to<umat>::from(inputMat);
    tmpMat.reshape(28,28);
    for(int i=0;i<28;++i){
        for(int j=0;j<28;++j)
            cout<<as_scalar(tmpMat.at(i,j))<<" ";
        cout<<endl;
    }
    //cout<<endl<<"Y=: "<<as_scalar(Y.row(200))<<endl;
    inputMat.reshape(1,784);
    return output(inputMat);
}

mat network::backpropogate(mat Inputs, mat Outputs){
    mat Theta1 = reshape(m_nn_params.rows(0,(m_inputLayerSize+1)*(m_hiddenLayerSize)-1),m_hiddenLayerSize,m_inputLayerSize+1);
    mat Theta2 = reshape(m_nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSize),m_nn_params.n_rows-1), m_outputLayerSize, m_hiddenLayerSize+1);
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
    cost += (accu(square(Theta1.cols(1,Theta1.n_cols-1)))+accu(square(Theta2.cols(1,m_hiddenLayerSize))))*m_lambda/(2*InputSize);
    cout<<"\t\tCost (regularized) = "<<cost<<endl;
    Theta1_grad /= InputSize;
    Theta2_grad /= InputSize;

    Theta1_grad += join_horiz(zeros<mat>(Theta1.n_rows,1), (m_lambda/InputSize)*Theta1.cols(1,Theta1.n_cols-1));
    Theta2_grad += join_horiz(zeros<mat>(Theta2.n_rows,1), (m_lambda/InputSize)*Theta2.cols(1,Theta2.n_cols-1));

    mat grad = join_vert(vectorise(Theta1_grad),vectorise(Theta2_grad));
    //nn_params -= alpha*grad;
    return grad;
}

void network::train(){
    int Total = m_X.n_rows, batch_size = m_X.n_rows/100;
    cout<<"\n\tStarting batch training.\n\n";

    cout<<"Prediction Accuracy before training: "<<accuracy(m_X, m_Y)<<endl<<endl;

    for(int k = 0;k<Total/batch_size; ++k){
        mat X_batch = m_X.rows(batch_size*(k),batch_size*(k+1)-1);
        mat Y_batch = m_Y.rows(batch_size*(k),batch_size*(k+1)-1);
        cout<<"Batch "<<k+1<<endl;
        for(int i=0; i<15; ++i){
            cout<<"\tIteration "<<i<<endl;
            m_nn_params = m_mu*m_nn_params - backpropogate(X_batch, Y_batch);
        }
    }
    m_Theta1 = reshape(m_nn_params.rows(0,(m_inputLayerSize+1)*(m_hiddenLayerSize)-1),m_hiddenLayerSize,m_inputLayerSize+1);
    m_Theta2 = reshape(m_nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSize),m_nn_params.size()-1), m_outputLayerSize, m_hiddenLayerSize+1);

    cout<<"Prediction Accuracy on training set: "<<accuracy(m_X, m_Y);
    cout<<"\n\nUsing test set"<<endl;
    load(inpPath);

    cout<<"Prediction Accuracy on test set: "<<accuracy(m_X, m_Y);
    mat hyper_params = {m_inputLayerSize,m_hiddenLayerSize,m_outputLayerSize};
    m_nn_params = join_vert(vectorise(hyper_params),m_nn_params);
    m_nn_params.save("parameters.csv",csv_ascii);
}

void network::load(string path, int startInd, int endInd){
    inpPath = path;
    m_X.load(path);
    m_X = shuffle(m_X);
    if(endInd)
        m_X = m_X.rows(startInd,endInd);
    else
        m_X = m_X.rows(startInd,m_X.n_rows-1);
    m_Y = m_X.col(0);
    m_X = m_X.cols(1,m_X.n_cols-1);
}

double network::accuracy(mat &X, mat &Y){
    umat prediction = (predict(X)==Y);
    return as_scalar(accu(prediction)*100.0/prediction.n_elem);
}

#endif // INTELLIDETECT_H_INCLUDED
