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

#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>
using namespace std;
namespace Property {
    const string hiddenLayerCount = "hiddenLayerCount";
    const string saveLocation = "saveLocation";
    const string Id = "Id";
    namespace layers {
        const string inputLayerSize = "layers.inputLayerSize";
        const string outputLayerSize = "layers.outputLayerSize";
        const string hiddenLayerSize(int num){
            return "layers.hiddenLayerSize"+to_string(num);
        }
    }
    namespace hyperParameters {
        const string regularizationParameter = "hyperParameters.regularizationParameter";
        const string learningRate = "hyperParameters.learningRate";
        const string momentumConstant = "hyperParameters.momentumConstant";
        const string iterCount = "hyperParameters.iterCount";
        const string batchSize = "hyperParameters.batchSize";
        const string numEpochs = "hyperParameters.numEpochs";
    }
}

#endif // CONSTANTS_H
