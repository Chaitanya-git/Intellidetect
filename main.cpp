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

int main(){
    cout<<"ANN Test Using the MNIST dataset"<<endl;
    network* net = new network();
    //net->load("/home/chaitanya/Downloads/MNIST-dataset-in-different-formats/data/CSV format/mnist_train.csv", 0, 9999);
    //net->train();
    net->predict("/home/chaitanya/Downloads/6.pgm").print();
    return 0;
}
