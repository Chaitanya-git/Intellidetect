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
