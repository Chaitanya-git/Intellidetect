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
        string inpPath;
    public:
        network(int, int, int);
        mat activation(mat);
        mat activationGradient(mat);
        mat randInitWeights(int, int);
        mat predict(mat);
        mat predict(string);
        mat output(string);
        mat output(mat);
        mat backpropogate(mat, mat, double);
        void train(double, double, double);
        void load(string, int, int);
        double accuracy(mat&, mat&);
};

network::network(int inpSize = 784, int HdSize = 100, int OpSize = 10){
    m_inputLayerSize = inpSize;
    m_hiddenLayerSize = HdSize;
    m_outputLayerSize = OpSize;
    if(m_nn_params.load("parameters.csv")==false){
        cout<<"Randomly initialising weights."<<endl;
        m_Theta1 = randInitWeights(m_hiddenLayerSize, m_inputLayerSize+1);
        m_Theta2 = randInitWeights(m_outputLayerSize,m_hiddenLayerSize+1);
        m_nn_params = join_vert(vectorise(m_Theta1),vectorise(m_Theta2)); //the weights in a more manageable format
    }
    else{
        cout<<"Loading Network sizes from file."<<endl;
        m_inputLayerSize = as_scalar(m_nn_params(0));
        m_hiddenLayerSize = as_scalar(m_nn_params(1));
        m_outputLayerSize = as_scalar(m_nn_params(2));
        m_nn_params = m_nn_params.rows(3,m_nn_params.n_rows-1);
        m_Theta1 = reshape(m_nn_params.rows(0,(m_inputLayerSize+1)*(m_hiddenLayerSize)-1),m_hiddenLayerSize,m_inputLayerSize+1);
        m_Theta2 = reshape(m_nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSize),m_nn_params.size()-1), m_outputLayerSize, m_hiddenLayerSize+1);
    }
    cout<<"Weights Initialized"<<endl;
}

void network::load(string path, int startInd=0, int endInd=0){
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

mat network::activation(mat z){
    return 1.0/(1.0 + exp(-z)); //Sigmoid Function
}

mat network::activationGradient(mat z){ //Derivative of the sigmoid function
    mat g = activation(z);
    return g%(1-g);         // '%'is the overloaded element wise multiplication operator.
}

mat network::randInitWeights(int Lin, int Lout){
    double epsilon = 0.52;
    return randu(Lin,Lout)*(2*epsilon) - epsilon;
}

mat network::predict(mat Input){
    int InputSize = Input.n_rows;
    Input = join_horiz(ones<mat>(InputSize,1),Input);
    mat z2 = Input*trans(m_Theta1);
    mat h1 = join_horiz(ones<mat>(z2.n_rows,1),activation(z2));
    mat h2 = activation(h1*trans(m_Theta2));
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
    mat h1 = join_horiz(ones<mat>(z2.n_rows,1),activation(z2));
    mat h2 = activation(h1*trans(m_Theta2));
    mat pred = zeros(InputSize,1);
    cout<<"Output: "<<h2<<endl;
    cout<<"Sum of outputs"<<accu(h2)<<endl;
    for(int i=0; i<InputSize; ++i){
        pred(i) = h2.row(i).max();
    }
    return pred;

}
mat network::predict(string input){
    mat inputMat;
    inputMat.load(input);
    cout<<inputMat.size()<<endl;
    umat tmpMat = conv_to<umat>::from(inputMat);
    tmpMat.reshape(28,28);
    for(int i=0;i<28;++i){
        for(int j=0;j<28;++j)
            cout<<as_scalar(tmpMat.at(i,j))<<" ";
        cout<<endl;
    }
    inputMat.reshape(1,784);
    return predict(inputMat);
}

mat network::output(string input){
    mat inputMat;
    inputMat.load(input);
    cout<<inputMat.size()<<endl;
    umat tmpMat = conv_to<umat>::from(inputMat);
    tmpMat.reshape(28,28);
    for(int i=0;i<28;++i){
        for(int j=0;j<28;++j)
            cout<<as_scalar(tmpMat.at(i,j))<<" ";
        cout<<endl;
    }
    inputMat.reshape(1,784);
    return output(inputMat);
}

mat network::backpropogate(mat Inputs, mat Outputs, double lambda){
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
        mat a2 = activation(z2);
        a2 = join_vert(ones<mat>(1,1),a2);
        mat z3 = Theta2*a2;
        mat a3 = activation(z3);
        output_tmp(as_scalar(Outputs(i))) = 1;

        cost += as_scalar(accu(output_tmp%log(a3)+(1-output_tmp)%log(1-a3)))/InputSize*(-1);

        mat delta_3 = a3-output_tmp;
        mat delta_2 = trans(Theta2.cols(1,Theta2.n_cols-1))*delta_3%activationGradient(z2);

        Theta1_grad += delta_2*CurrentInput.t();
        Theta2_grad += delta_3*a2.t();
        output_tmp(as_scalar(Outputs(i)),0) = 0;
    }
    cout<<"\tCost(unregularized) = "<<cost;
    cost += (accu(square(Theta1.cols(1,Theta1.n_cols-1)))+accu(square(Theta2.cols(1,m_hiddenLayerSize))))*lambda/(2*InputSize);
    cout<<"\t\tCost (regularized) = "<<cost<<endl;
    Theta1_grad /= InputSize;
    Theta2_grad /= InputSize;

    Theta1_grad += join_horiz(zeros<mat>(Theta1.n_rows,1), (lambda/InputSize)*Theta1.cols(1,Theta1.n_cols-1));
    Theta2_grad += join_horiz(zeros<mat>(Theta2.n_rows,1), (lambda/InputSize)*Theta2.cols(1,Theta2.n_cols-1));

    mat grad = join_vert(vectorise(Theta1_grad),vectorise(Theta2_grad));
    return grad;
}

void network::train(double lambda = 0.5,double alpha = 0.05,double mu = 1){//regularization parameter and learning rate and a momentum constant
    int Total = m_X.n_rows, batch_size = m_X.n_rows/100;
    cout<<"\n\tStarting batch training.\n\n";

    cout<<"Prediction Accuracy before training: "<<accuracy(m_X, m_Y)<<endl<<endl;

    for(int k = 0;k<Total/batch_size; ++k){
        mat X_batch = m_X.rows(batch_size*(k),batch_size*(k+1)-1);
        mat Y_batch = m_Y.rows(batch_size*(k),batch_size*(k+1)-1);
        cout<<"Batch "<<k+1<<endl;
        for(int i=0; i<35; ++i){
            cout<<"\tIteration "<<i+1<<endl;
            m_nn_params = mu*m_nn_params - alpha*backpropogate(X_batch, Y_batch, lambda);
        }
    }
    m_Theta1 = reshape(m_nn_params.rows(0,(m_inputLayerSize+1)*(m_hiddenLayerSize)-1),m_hiddenLayerSize,m_inputLayerSize+1);
    m_Theta2 = reshape(m_nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSize),m_nn_params.size()-1), m_outputLayerSize, m_hiddenLayerSize+1);

    cout<<"Prediction Accuracy on training set: "<<accuracy(m_X, m_Y);
    cout<<"\n\nUsing test set"<<endl;
    load(inpPath);

    cout<<"Prediction Accuracy on test set: "<<accuracy(m_X, m_Y)<<endl;
    mat hyper_params = {
                        static_cast<double>(m_inputLayerSize),
                        static_cast<double>(m_hiddenLayerSize),
                        static_cast<double>(m_outputLayerSize)
                       };
    mat tmp_params = join_vert(vectorise(hyper_params),m_nn_params);
    tmp_params.save("parameters3.csv",csv_ascii);
}

double network::accuracy(mat &X, mat &Y){
    umat prediction = (predict(X)==Y);
    return as_scalar(accu(prediction)*100.0/prediction.n_elem);
}

class ReLU :public network {
    public:
        mat activation(mat z);
        mat activationGradient(mat z);
        mat randInitWeights(int Lin, int Lout);
};

mat ReLU::activation(mat z){
    mat act = zeros(z.n_rows,1);
    for(unsigned int i=0;i<z.n_rows;++i){
        if(z(i,0)>0)
            act(i,1) = z(i,1);
    }
    return act;
}

mat ReLU::activationGradient(mat z){
    mat grad = zeros(z.n_rows,1);
    for(unsigned int i=0;i<z.n_rows;++i){
        if(z(i,0)>0)
            grad(i,0) = 1;
    }
    return grad;
}

mat ReLU::randInitWeights(int Lin, int Lout){
    double epsilon = 0.99;
    return randu(Lin,Lout)*(2*epsilon) - epsilon;
}
#endif // INTELLIDETECT_H_INCLUDED
