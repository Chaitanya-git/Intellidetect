#include <iostream>
#include <armadillo>
#include "IntelliDetect.h"

using namespace std;

int main(){
    cout<<"ANN Debugging"<<endl;
    cout<<"\n\nSigmoid Function test:"<<endl;
    mat sig = {{100000, -100000, 0}};
    cout<<"sigmoid(100000, -100000, 0) = "<<sigmoid(sig);
    mat grad = {{1, -0.5 ,0, 0.5, 1}};
    cout<<"\n\nSigmoid gradient function test:"<<endl
        <<"sigmoidGradient(1, -0.5 ,0, 0.5, 1) = "<<sigmoidGradient(grad);

    cout<<"Weights initializing test:"<<endl;
    cout<<"Weights:\n"<<randInitWeights(3,4)<<endl;

    return 0;
}
