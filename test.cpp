/*
 * Copyright (C) 2017 Chaitanya and Geeve George
 * This file is part of Intellidetect.
 *
 *  Intellidetect is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Intellidetect is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Intellidetect.  If not, see <http://www.gnu.org/licenses/>.
 */
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
