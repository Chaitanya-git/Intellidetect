#ifndef INTELLIDETECT_H_INCLUDED
#define INTELLIDETECT_H_INCLUDED
#include <armadillo>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <ctime>
#include <sys/stat.h>
#include <proptree.h>
#define INTELLI_VERSION "2.2.2"

/* This header file contains definitions for functions for handling various ANN processes */

using namespace arma;
using namespace std;

namespace IntelliDetect{

    mat sigmoid(mat z){
        return 1.0/(1.0 + exp(-z));
    }

    mat sigmoidGradient(mat z){
        mat g = sigmoid(z);
        return g%(1-g);         // '%'is the overloaded element wise multiplication operator.
    }

    mat RectifiedLinearUnitActivation(mat z){
        return (z+abs(z))/2;
    }

    mat RectifiedLinearUnitActivationGradient(mat z){
        return (z+abs(z))/(2*z);
    }

    mat randWeights(int Lin, int Lout){
        const double epsilon = 0.52;
        return randu(Lin,Lout)*(2*epsilon) - epsilon;
    }

    class network{
            mat m_Inputs,m_Lables;
            vector<mat> m_Theta;
            int m_inputLayerSize,m_outputLayerSize;
            vector<int> m_hiddenLayerSizes;
            string m_paramPath;
            vector<string> m_inpPaths;
            mat (*activation)(mat);
            mat (*activationGradient)(mat);
            propertyTree properties;
            bool initializeFromPropertyTree();
            void randInitWeights();
            bool constructWeightsFromParameters(mat&);
            void constructParameters(string);
            bool checkLayerSizes();
            void buildPropertyTree();
            vector<mat> forwardPass(mat);

        public:

            vector<long double> trainSetCostsReg;
            vector<long double> trainSetCosts;
            network(mat (*activationPtr)(mat), mat (*activationGradientPtr)(mat),string, int, int, int);
            network(string,mat (*activationPtr)(mat), mat (*activationGradientPtr)(mat));
            network(propertyTree,mat (*activationPtr)(mat), mat (*activationGradientPtr)(mat));
            mat predict(mat);
            mat predict(string);
            mat output(string);
            mat output(mat);
            void backpropogate(mat, mat, double, double);
            mat numericalGradient(mat);
            void train(double, double, double);
            void train(string, int, double, double, double);
            bool load(vector<string>, int, int);
            bool load(string, int, int);
            bool load(mat&,mat&);
            bool save(string);
            double accuracy(mat&, mat&);
            string getPath();
    };

    mat network::numericalGradient(mat z){
        double h = 1e-10;
        return (activation(z+h)-activation(z))/h;
    }

    bool network::initializeFromPropertyTree(){
        bool status = true;
        int hiddenLayerCount = 0;
        if(properties.isSet(Property::hiddenLayerCount)){
            hiddenLayerCount = stoi(properties.getProperty(Property::hiddenLayerCount));
            m_hiddenLayerSizes.reserve(hiddenLayerCount);
        }
        else status = false;
        if(properties.isSet(Property::layers::inputLayerSize))
            m_inputLayerSize = stoi(properties.getProperty(Property::layers::inputLayerSize));
        else status = false;

        for(int i=0;i<hiddenLayerCount;++i){
            if(properties.isSet(Property::layers::hiddenLayerSize(i)))
                m_hiddenLayerSizes[i] = stoi(properties.getProperty(Property::layers::hiddenLayerSize(i)));
            else status = false;
        }

        if(properties.isSet(Property::layers::outputLayerSize))
            m_outputLayerSize = stoi(properties.getProperty(Property::layers::outputLayerSize));
        else status = false;
        if(properties.isSet(Property::saveLocation))
            m_paramPath = properties.getProperty(Property::saveLocation);
        return status;
    }

    void network::buildPropertyTree(){
        properties.data = string("Version ")+string(INTELLI_VERSION);
        properties.setPropertyIfNotSet(Property::Id,"TestNetwork");
        properties.setPropertyIfNotSet(Property::hiddenLayerCount,to_string(m_hiddenLayerSizes.capacity()));
        properties.setPropertyIfNotSet(Property::layers::inputLayerSize,to_string(m_inputLayerSize));
        for(unsigned int i=0;i<m_hiddenLayerSizes.capacity();++i)
            properties.setPropertyIfNotSet(Property::layers::hiddenLayerSize(i),to_string(m_hiddenLayerSizes[i]));
        properties.setPropertyIfNotSet(Property::layers::outputLayerSize,to_string(m_outputLayerSize));
        properties.setPropertyIfNotSet(Property::saveLocation,m_paramPath);
    }

    network::network(mat (*activationPtr)(mat) = sigmoid, mat (*activationGradientPtr)(mat) = sigmoidGradient, string param = "", int inpSize = 784, int HdSize = 100, int OpSize = 10){
        m_paramPath = param;
        m_inputLayerSize = inpSize;
        m_hiddenLayerSizes.reserve(1);
        m_hiddenLayerSizes[0] = HdSize;
        m_outputLayerSize = OpSize;
        activation = activationPtr;
        activationGradient = activationGradientPtr;
        buildPropertyTree();
        constructParameters(param);
        trainSetCosts.resize(0);
        trainSetCostsReg.resize(0);
    }

    network::network(string path,mat (*activationPtr)(mat) = sigmoid, mat (*activationGradientPtr)(mat) = sigmoidGradient){
        if(path.back()!='/')
            path.append("/",1);
        activation = activationPtr;
        activationGradient = activationGradientPtr;
        properties.load(path+string("network.conf"));
        properties.setProperty(Property::saveLocation,path);

        bool status = initializeFromPropertyTree();
        if(!status)
            cout<<"Error: provided configuration is incomplete"<<endl;

        constructParameters(path+string("parameters.csv"));

        fstream trainStat;
        trainStat.open(path+string("trainingStats.csv"),ios::in);
        if(!trainStat){
            trainStat.close();
            return;
        }
        double trainSetCost,trainSetCostReg;
        char ch =' ';
        while(!(trainStat>>trainSetCost>>ch>>trainSetCostReg>>ch).eof()){
            trainSetCosts.push_back(trainSetCost);
            trainSetCostsReg.push_back(trainSetCostReg);
        }

        trainStat.close();
    }

    network::network(propertyTree props,mat (*activationPtr)(mat) = sigmoid, mat (*activationGradientPtr)(mat) = sigmoidGradient){
        activation = activationPtr;
        activationGradient = activationGradientPtr;
        properties = props;
        bool status = initializeFromPropertyTree();
        constructParameters("");
        if(!status)
            cout<<"Error: Provided property tree is incomplete"<<endl;
    }

    bool network::checkLayerSizes(){
        bool layerSizesSet = true;
        if(!(properties.isSet(Property::layers::inputLayerSize) &&
           properties.isSet(Property::layers::outputLayerSize)))
            layerSizesSet = false;
        int hiddenLayerCount = 0;
        if(properties.isSet(Property::hiddenLayerCount))
            hiddenLayerCount = stoi(properties.getProperty(Property::hiddenLayerCount));
        else layerSizesSet = false;
        for(int i=0;i<hiddenLayerCount;++i)
            if(!properties.isSet(Property::layers::hiddenLayerSize(i)))
                layerSizesSet=false;
        return layerSizesSet;
    }

    void network::randInitWeights(){
        if(checkLayerSizes()){
            m_Theta[0] = randWeights(m_hiddenLayerSizes[0], m_inputLayerSize+1);
            m_Theta[1] = randWeights(m_outputLayerSize,m_hiddenLayerSizes[0]+1);
        }
        else cout<<"Error: trying to initialize weights without setting layer sizes"<<endl;
    }

    void network::constructParameters(string path){
        mat nn_params;
        m_Theta = vector<mat>(2);
        if(!nn_params.load(path)){
            cout<<"Randomly initialising weights."<<endl;
            randInitWeights();
        }
        else{
            cout<<"Loading Network sizes from file."<<endl;
            constructWeightsFromParameters(nn_params);
        }
        cout<<"Weights Initialized"<<endl;
    }

    bool network::constructWeightsFromParameters(mat &nn_params){
        bool layerSizesSet = checkLayerSizes();
        //Problem: checkLayerSizes checks with propTree, but there's no way to know if variables have been initialized with proptree values
        //Possible fix: Stop using seperate variables. Directly fetch values from propTree.
        bool status = true;
        int hiddenLayerCount = m_hiddenLayerSizes.capacity();
        if(!layerSizesSet){
            m_inputLayerSize = as_scalar(nn_params(0));
            for(int i=0;i<hiddenLayerCount;++i)
                m_hiddenLayerSizes[i] = as_scalar(nn_params(i+1));
            m_outputLayerSize = as_scalar(nn_params(hiddenLayerCount+1));
            buildPropertyTree();
        }
        for(int i=0;i<hiddenLayerCount;++i)
            if(!(m_hiddenLayerSizes[i] == as_scalar(nn_params(i+1))))
                status = false;
        if(!(m_inputLayerSize == as_scalar(nn_params(0)) && m_outputLayerSize == as_scalar(nn_params(hiddenLayerCount+1))))
            status = false;
        if(status){
            nn_params = nn_params.rows(hiddenLayerCount+1,nn_params.n_rows-1);
            m_Theta[0] = reshape(nn_params.rows(0,(m_inputLayerSize+1)*(m_hiddenLayerSizes[0])-1),m_hiddenLayerSizes[0],m_inputLayerSize+1);
            m_Theta[1] = reshape(nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSizes[0]),nn_params.size()-1), m_outputLayerSize, m_hiddenLayerSizes[0]+1);
        }
        else{
            if(properties.isSet(Property::saveLocation))
                nn_params.save(properties.getProperty(Property::saveLocation)+string("parameters.csv.back"));
            randInitWeights();
        }

        cout<<"Set layer sizes to "<<m_inputLayerSize<<", ";
        for(auto i: m_hiddenLayerSizes)
            cout<<i<<", ";
        cout<<m_outputLayerSize<<endl;

        return status;
    }

    bool network::load(vector<string> paths, int startInd=0, int endInd=0){
        if(!paths.size()){
            cout<<"network::load(): error: Input file list is empty.";
            return false;
        }

        m_inpPaths = paths;
        mat tmpX;
        m_Inputs.resize(0,0);
        unsigned int i=0;
        do{
            tmpX.load(m_inpPaths.at(i++));
            m_Inputs = join_vert(m_Inputs,tmpX);
        }while(i<m_inpPaths.size()-1); //last path represents test set.)
        m_Inputs = shuffle(m_Inputs);
        if(endInd)
            m_Inputs = m_Inputs.rows(startInd,endInd);
        else
            m_Inputs = m_Inputs.rows(startInd,m_Inputs.n_rows-1);
        m_Lables = m_Inputs.col(0);
        m_Inputs = m_Inputs.cols(1,m_Inputs.n_cols-1);

        return true;
    }

    bool network::load(string path, int startInd=0, int endInd=0){
        if(m_Inputs.load(path)==false)
            return false;
        m_Inputs = shuffle(m_Inputs);
        if(endInd)
            m_Inputs = m_Inputs.rows(startInd,endInd);
        else
            m_Inputs = m_Inputs.rows(startInd,m_Inputs.n_rows-1);
        m_Lables = m_Inputs.col(0);
        m_Inputs = m_Inputs.cols(1,m_Inputs.n_cols-1);

        return true;
    }

    bool network::load(mat &Inputs, mat &Labels){
        if(Inputs.n_rows != Labels.n_rows){
            cout<<"network::load() error: Number of inputs do not match number of labels";
            return false;
        }
        if(Labels.n_cols>1){
            cout<<"network::load() error: Labels cannot be a vector";
            return false;
        }
        m_Inputs = Inputs;
        m_Lables = Labels;

        return true;
    }

    bool network::save(string path){
        //Construct path
        string fullpath("");
        fullpath += path;
        if(path.back()!='/')
            fullpath.append("/",1);
        string folderName;
        time_t Time;
        if(properties.isSet(Property::Id))
            folderName = string(properties.getProperty(Property::Id));
        else{
            time(&Time);
            folderName = string("network ")+string(asctime(localtime(&Time)));
        }
        fullpath += folderName+string("/");
        mkdir(fullpath.c_str(),0755);
        fstream trainStat;

        //Save training stats
        trainStat.open(fullpath+string("trainingStats.csv"),ios::out);
        for(unsigned int i=0;i<trainSetCosts.size();++i){
            trainStat<<trainSetCosts[i]<<", "<<trainSetCostsReg[i]<<", "<<endl;
        }
        trainStat.close();

        //Save weights
        int hiddenLayerCount = m_hiddenLayerSizes.capacity();
        mat hyper_params = zeros<mat>(1,hiddenLayerCount+2);
        hyper_params.at(0) =  static_cast<double>(m_inputLayerSize);

        for(int i=0;i<hiddenLayerCount;++i)
            hyper_params.at(i) = static_cast<double>(m_hiddenLayerSizes[i]);

        hyper_params.at(hiddenLayerCount+1) = static_cast<double>(m_outputLayerSize);

        mat nn_params = join_vert(vectorise(m_Theta[0]),vectorise(m_Theta[1]));
        mat tmp_params = join_vert(vectorise(hyper_params),nn_params);
        tmp_params.save((fullpath+string("parameters.csv")),csv_ascii);

        //Save propertyTree
        fstream configFile;
        configFile.open(fullpath+string("network.conf"),ios::out);
        configFile<<string("//Intellidetect Configuration file\n");
        configFile<<properties.toString();
        configFile.close();
        return true;
    }
    vector<mat> network::forwardPass(mat Input){
        vector<mat> layers;
        layers.reserve(m_hiddenLayerSizes.capacity()+1);
        int InputSize = Input.n_rows;
        Input = join_horiz(ones<mat>(InputSize,1),Input);
        layers.push_back(Input*trans(m_Theta[0]));
        mat tmp = join_horiz(ones<mat>(layers[0].n_rows,1),activation(layers[0]));
        layers.push_back(sigmoid(tmp*trans(m_Theta[1])));
        return layers;
    }

    mat network::predict(mat Input){
        vector<mat> layers = forwardPass(Input);
        int InputSize = Input.n_rows;
        mat pred = zeros(InputSize,1);
        for(int i=0; i<InputSize; ++i){
            pred(i) = layers.back().row(i).index_max();
        }
        return pred;
    }
    mat network::output(mat Input){
        int InputSize = Input.n_rows;
        vector<mat> layers = forwardPass(Input);
        cout<<"Output: "<<layers.back()<<endl;
        cout<<"Sum of outputs"<<accu(layers.back())<<endl;
        mat pred = zeros(InputSize,1);
        for(int i=0; i<InputSize; ++i){
            pred(i) = layers.back().row(i).max();
        }
        return pred;

    }
    mat network::predict(string input){
        mat inputMat;
        inputMat.load(input);
        inputMat.reshape(1,784);
        return predict(inputMat);
    }

    mat network::output(string input){
        mat inputMat;
        inputMat.load(input);
        inputMat.reshape(1,784);
        return output(inputMat);
    }

    void network::backpropogate(mat Inputs, mat Outputs, double regularizationParameter, double learningRate){
        int InputSize = Inputs.n_rows;
        long double cost = 0;
        mat Theta1_grad = zeros<mat>(size(m_Theta[0]));
        mat Theta2_grad = zeros<mat>(size(m_Theta[1]));
        Inputs = join_horiz(ones<mat>(InputSize,1), Inputs); //Add the weights from the bias neuron.
        mat output_tmp = zeros<mat>(10,1);
        for(int i=0; i<InputSize; ++i){
            vector<mat> layers = forwardPass(Inputs.row(i).cols(0,Inputs.n_cols-2));
            mat activation3 = sigmoid(layers.back().t());
            mat activation2 = join_horiz(ones<mat>(1,1),sigmoid(layers[0]));
            output_tmp(as_scalar(Outputs(i))) = 1;

            cost += as_scalar(accu(output_tmp%log(activation3)+(1-output_tmp)%log(1-activation3)))/InputSize*(-1);

            mat delta_3 = activation3-output_tmp;
            mat delta_2 = trans(m_Theta[1].cols(1,m_Theta[1].n_cols-1))*delta_3%activationGradient(layers[0].t());
            //mat delta_2_check = trans(m_Theta2.cols(1,m_Theta2.n_cols-1))*delta_3%numericalGradient(z2);
            //cout<<"Diff in activation grad and num_grad: "<<accu(delta_2-delta_2_check)<<endl;
            Theta1_grad += delta_2*Inputs.row(i);
            Theta2_grad += delta_3*activation2;
            output_tmp(as_scalar(Outputs(i)),0) = 0;
        }
        cout<<"\tCost(unregularized) = "<<cost;
        trainSetCosts.push_back(cost);
        cost += accu(square(m_Theta[0].cols(1,m_Theta[0].n_cols-1)))*regularizationParameter/(2.0*InputSize);

        //Put in loop
        cost += accu(square(m_Theta[1].cols(1,m_hiddenLayerSizes[0])))*regularizationParameter/(2.0*InputSize);

        cout<<"\t\tCost (regularized) = "<<cost<<endl;
        trainSetCostsReg.push_back(cost);
        Theta1_grad /= InputSize;
        Theta2_grad /= InputSize;

        Theta1_grad += join_horiz(zeros<mat>(m_Theta[0].n_rows,1), (regularizationParameter/InputSize)*m_Theta[0].cols(1,m_Theta[0].n_cols-1));
        Theta2_grad += join_horiz(zeros<mat>(m_Theta[1].n_rows,1), (regularizationParameter/InputSize)*m_Theta[1].cols(1,m_Theta[1].n_cols-1));

        m_Theta[0] -= learningRate*Theta1_grad;
        m_Theta[1] -= learningRate*Theta2_grad;
    }

    void network::train(double regularizationParameter = 0.25,double learningRate = 0.05,double momentumConstant = 1){
        properties.setProperty(Property::hyperParameters::regularizationParameter,to_string(regularizationParameter));
        properties.setProperty(Property::hyperParameters::learningRate,to_string(learningRate));
        properties.setProperty(Property::hyperParameters::momentumConstant,to_string(momentumConstant));
        int Total = m_Inputs.n_rows, batch_size = m_Inputs.n_rows/100, IterCnt = 50;
        if(!batch_size){
            batch_size = 1;
            IterCnt = 10;
        }
        cout<<"\n\tStarting batch training.\n\n";

        cout<<"Prediction Accuracy before training: "<<accuracy(m_Inputs, m_Lables)<<endl<<endl;
        double acc;
        do{
            for(int k = 0;k<Total/batch_size; ++k){
                mat Input_batch = m_Inputs.rows(batch_size*(k),batch_size*(k+1)-1);
                mat Label_batch = m_Lables.rows(batch_size*(k),batch_size*(k+1)-1);
                cout<<"Batch "<<k+1<<endl;
                for(int i=0; i<IterCnt; ++i){
                    cout<<"\tIteration "<<i+1<<endl;
                    backpropogate(Input_batch, Label_batch, regularizationParameter, learningRate);
                }
            }
            acc = accuracy(m_Inputs, m_Lables);
            cout<<"Prediction Accuracy on training set: "<<acc<<endl;
        }while(acc<70);

        if(m_inpPaths.size()){
            if(load(m_inpPaths.at(m_inpPaths.size()-1))==true){
                cout<<"\n\nUsing test set"<<endl;
                cout<<"Prediction Accuracy on test set: "<<accuracy(m_Inputs, m_Lables)<<endl;
            }
            else
                cout<<"Could not load test set. Training may be incomplete."<<endl;
        }
        else
            cout<<"No training set provided."<<endl;
        save(m_paramPath);
    }

    void network::train(string input, int label, double regularizationParameter = 0.5,double learningRate = 0.05,double momentumConstant = 1){//regularization parameter and learning rate and a momentum constant
        properties.setProperty(Property::hyperParameters::regularizationParameter,to_string(regularizationParameter));
        properties.setProperty(Property::hyperParameters::learningRate,to_string(learningRate));
        properties.setProperty(Property::hyperParameters::momentumConstant,to_string(momentumConstant));

        cout<<"\n\tStarting individual training.\n\n";

        mat Input = zeros(28,28);
        mat Label = zeros(1,1);
        Input.load(input);
        Input.reshape(1,784);
        Label.at(0) = label;
        cout<<"Overall Prediction Accuracy before training: "<<accuracy(Input, Label)<<endl<<endl;

        int i=0;
        while(1){
            cout<<"\tIteration "<<++i<<endl;
            backpropogate(Input, Label, regularizationParameter, learningRate);
            mat out = predict(Input);
            mat conf = output(Input);
            cout<<"out.at(0) = "<<out.at(0)<<endl<<"confidence = "<<conf(0)<<endl;
            if(out.at(0) == label && conf(0)>0.5)
                break;
        }

        cout<<"Done training on given example."<<endl;
        cout<<"Overall Prediction Accuracy after training: "<<accuracy(Input, Label)<<endl<<endl;
        save(m_paramPath);
    }

    double network::accuracy(mat &X, mat &Y){
        umat prediction = (predict(X)==Y);
        return as_scalar(accu(prediction)*100.0/prediction.n_elem);
    }

    string network::getPath(){
        return m_paramPath;
    }

    class ConvNet :public network{
            mat m_Inputs,m_Lables;
            int m_imgHt,m_imgWth, m_strideLn;
            vector<int> m_convSizes;
            network m_FCnet;
            vector<mat> m_kernels;
            mat pool(mat,int);
        public:
            ConvNet(int,int,int,vector<int>,vector<int>);
            mat backpropogate(mat, mat, double);
            mat predict(mat);
            mat output(mat);
    };

    ConvNet::ConvNet(int imgHt, int imgWth, int strideLn, vector<int> kernelSize, vector<int> convSizes){
        m_FCnet = network(sigmoid, sigmoidGradient, "", 50, 60, 10);
        m_imgHt = imgHt;
        m_imgWth = imgWth;
        m_strideLn = strideLn;
        m_convSizes = convSizes;
        for(unsigned int i=0;i<m_convSizes.size();++i){
            for(int j=0;j<m_convSizes[i];++j)
                m_kernels.push_back(ones(kernelSize[i],kernelSize[i]));
        }
    }

    mat ConvNet::pool(mat layer, int downsamplingFactor){
        mat linearLayer = vectorise(layer);
        mat downsampledLayer = zeros(linearLayer.n_rows/downsamplingFactor);
        for(unsigned int i=0;i<downsampledLayer.n_rows;++i){
            downsampledLayer(i,0) = linearLayer.rows(i*downsamplingFactor,(i+1)*downsamplingFactor-1).max();
        }
        return downsampledLayer;
    }
}
#endif // INTELLIDETECT_H_INCLUDED
