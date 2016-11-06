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
#include <stdexcept>
#define INTELLI_VERSION "2.2.2"

/* This header file contains definitions for functions for handling various ANN processes */

using namespace arma;
using namespace std;

namespace IntelliDetect{

    class invalidConfigException: public exception{
        virtual const char* what() const throw()
          {
            return "Provided configuration is incomplete";
          }
    };

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

    bool validatePropTree(propertyTree &properties){
        if(!properties.isSet(Property::layers::inputLayerSize)) return false;
        if(!properties.isSet(Property::layers::outputLayerSize)) return false;
        if(!properties.isSet(Property::hiddenLayerCount)) return false;
        int hiddenLayerCount = 0;
        hiddenLayerCount = stoi(properties.getProperty(Property::hiddenLayerCount));
        for(int i=0;i<hiddenLayerCount;++i)
            if(!properties.isSet(Property::layers::hiddenLayerSize(i))) return false;
        return true;
    }

    class network{
        protected:
            mat m_Inputs,m_Lables;
            vector<mat> m_Theta;
            int m_inputLayerSize,m_outputLayerSize;
            vector<int> m_hiddenLayerSizes;
            string m_paramPath;
            vector<string> m_inpPaths;
            mat (*activation)(mat);
            mat (*activationGradient)(mat);
            propertyTree properties;
            void initializeFromPropertyTree();
            void randInitWeights();
            bool constructWeightsFromParameters(mat&);
            void constructParameters(string);
            vector<mat> forwardPass(mat);

        public:

            vector<long double> trainSetCostsReg;
            vector<long double> trainSetCosts;
            network(string,mat (*activationPtr)(mat), mat (*activationGradientPtr)(mat));
            network(propertyTree,mat (*activationPtr)(mat), mat (*activationGradientPtr)(mat));
            mat predict(mat);
            mat predict(string);
            mat output(string);
            mat output(mat);
            vector<mat> backpropogate(mat, mat, double);
            mat numericalGradient(mat);
            void train();
            void train(string, int);
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

    void network::initializeFromPropertyTree(){
        int hiddenLayerCount = 0;
        hiddenLayerCount = stoi(properties.getProperty(Property::hiddenLayerCount));
        m_hiddenLayerSizes.reserve(hiddenLayerCount);
        m_inputLayerSize = stoi(properties.getProperty(Property::layers::inputLayerSize));

        for(int i=0;i<hiddenLayerCount;++i)
            m_hiddenLayerSizes[i] = stoi(properties.getProperty(Property::layers::hiddenLayerSize(i)));

        m_outputLayerSize = stoi(properties.getProperty(Property::layers::outputLayerSize));
        m_paramPath = properties.getProperty(Property::saveLocation);
    }

    network::network(string path,mat (*activationPtr)(mat) = sigmoid, mat (*activationGradientPtr)(mat) = sigmoidGradient){
        if(path.back()!='/')
            path.append("/",1);
        activation = activationPtr;
        activationGradient = activationGradientPtr;
        properties.load(path+string("network.conf"));
        int len = properties.getProperty(Property::Id).length();
        path.erase(path.length()-len-1);
        properties.setProperty(Property::saveLocation,path);

        if(!validatePropTree(properties))
            throw invalidConfigException();

        initializeFromPropertyTree();
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
        if(!validatePropTree(props))
            throw invalidConfigException();
        properties = props;
        initializeFromPropertyTree();
        constructParameters("");
    }

    void network::randInitWeights(){
        const int hiddenLayerCount = m_hiddenLayerSizes.capacity();
        m_Theta[0] = randWeights(m_hiddenLayerSizes[0], m_inputLayerSize+1);
        for(int i=1;i<hiddenLayerCount;++i)
            m_Theta[i] = randWeights(m_hiddenLayerSizes[i],m_hiddenLayerSizes[i-1]+1);
        m_Theta[hiddenLayerCount] = randWeights(m_outputLayerSize,m_hiddenLayerSizes[hiddenLayerCount-1]+1);
    }

    void network::constructParameters(string path){
        mat nn_params;
        m_Theta = vector<mat>(m_hiddenLayerSizes.capacity()+1);
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
        bool status = true;
        const int hiddenLayerCount = m_hiddenLayerSizes.capacity();

        //Check if sizes provided match with sizes stored with the parameters
        for(int i=0;i<hiddenLayerCount;++i)
            if(!(m_hiddenLayerSizes[i] == as_scalar(nn_params(i+1))))
                status = false;
        if(!(m_inputLayerSize == as_scalar(nn_params(0)) && m_outputLayerSize == as_scalar(nn_params(hiddenLayerCount+1))))
            status = false;

        if(status){
            nn_params = nn_params.rows(hiddenLayerCount+1,nn_params.n_rows-1);
            int prevSize = 0;
            int currentSize = (m_inputLayerSize+1)*(m_hiddenLayerSizes[0])-1;
            m_Theta[0] = reshape(nn_params.rows(prevSize,currentSize),m_hiddenLayerSizes[0],m_inputLayerSize+1);
            prevSize = currentSize+1;
            for(int i=1;i<hiddenLayerCount;++i){
                currentSize = (m_hiddenLayerSizes[i]-1)*m_hiddenLayerSizes[i];
                m_Theta[i] = reshape(nn_params.rows(prevSize,currentSize),m_hiddenLayerSizes[i],m_hiddenLayerSizes[i-1]+1);
                prevSize = currentSize+1;
            }
            m_Theta[hiddenLayerCount] = reshape(nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSizes[hiddenLayerCount-1]),nn_params.size()-1), m_outputLayerSize, m_hiddenLayerSizes[hiddenLayerCount-1]+1);
        }
        else{
            if(properties.isSet(Property::saveLocation))
                nn_params.save(properties.getProperty(Property::saveLocation)+string("parameters.csv.back"), csv_ascii);
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
            cout<<"network::load() error: Labels have to be a vector";
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
        mat hyper_params = zeros<mat>(hiddenLayerCount+2,1);
        hyper_params(0,0) =  m_inputLayerSize;

        for(int i=0;i<hiddenLayerCount;++i)
            hyper_params(i+1,0) = m_hiddenLayerSizes[i];

        hyper_params(hiddenLayerCount+1,0) = m_outputLayerSize;

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
        mat tmp;
        for(unsigned i=1;i<=m_hiddenLayerSizes.capacity();++i){
            tmp = join_horiz(ones<mat>(layers[i-1].n_rows,1),activation(layers[i-1]));
            layers.push_back(tmp*trans(m_Theta[i]));
        }
        layers[layers.capacity()-1] = sigmoid(layers.back());
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

    vector<mat> network::backpropogate(mat Inputs, mat Outputs, double regularizationParameter){
        int InputSize = Inputs.n_rows;
        long double cost = 0;

        vector<mat> Theta_grad(m_Theta.capacity());
        vector<mat> act(m_Theta.capacity());
        vector<mat> delta(m_Theta.capacity());

        for(unsigned i=0;i<m_Theta.capacity();++i)
            Theta_grad[i] = zeros<mat>(size(m_Theta[i]));

        Inputs = join_horiz(ones<mat>(InputSize,1), Inputs); //Add the weights from the bias neuron.
        mat output_tmp = zeros<mat>(m_outputLayerSize,1);

        for(int i=0; i<InputSize; ++i){
            vector<mat> layers = forwardPass(Inputs.row(i).cols(0,Inputs.n_cols-2));
            act[act.capacity()-1] = layers.back().t();
            for(unsigned i=0;i<act.capacity()-1;++i)
                act[i] = join_horiz(ones<mat>(1,1),sigmoid(layers[i]));
            output_tmp(as_scalar(Outputs(i))) = 1;

            cost += as_scalar(accu(output_tmp%log(act.back())+(1-output_tmp)%log(1-act.back())))/InputSize*(-1);

            delta[delta.capacity()-1] = act.back()-output_tmp;
            for(int i=delta.capacity()-2;i>=0;--i)
                delta[i] = trans(m_Theta[i+1].cols(1,m_Theta[i+1].n_cols-1))*delta[i+1]%activationGradient(layers[i].t());
            //mat delta_2_check = trans(m_Theta[1].cols(1,m_Theta[1].n_cols-1))*delta_3%numericalGradient(layers[0].t());
            //cout<<"Diff in activation grad and num_grad: "<<accu(delta_2-delta_2_check)<<endl;
            Theta_grad[0] += delta[0]*Inputs.row(i);
            for(unsigned i=1;i<Theta_grad.capacity();++i)
                Theta_grad[i] += delta[i]*act[i-1];
            output_tmp(as_scalar(Outputs(i)),0) = 0;
        }
        cout<<"\tCost(unregularized) = "<<cost;
        trainSetCosts.push_back(cost);

        for(unsigned i=0;i<m_Theta.capacity();++i){
            cost += accu(square(m_Theta[0].cols(1,m_Theta[0].n_cols-1)))*regularizationParameter/(2.0*InputSize); //Add regularization terms
            Theta_grad[i] /= InputSize;
            Theta_grad[i] += join_horiz(zeros<mat>(m_Theta[i].n_rows,1), (regularizationParameter/InputSize)*m_Theta[i].cols(1,m_Theta[i].n_cols-1)); //Add regularization terms
        }
        cout<<"\t\tCost (regularized) = "<<cost<<endl;
        trainSetCostsReg.push_back(cost);

        return Theta_grad;
    }

    void network::train(){
        double regularizationParameter = 0, learningRate = 0, momentumConstant = 0;
        int Total = m_Inputs.n_rows;
        int IterCnt = 0, batch_size = Total/100;
        //Get hyperParameters from property tree
        if(properties.isSet(Property::hyperParameters::regularizationParameter))
            regularizationParameter = stod(properties.getProperty(Property::hyperParameters::regularizationParameter));
        if(properties.isSet(Property::hyperParameters::learningRate))
            learningRate = stod(properties.getProperty(Property::hyperParameters::learningRate));
        else{
            cout<<properties.getProperty(Property::hyperParameters::learningRate);
            cout<<"No valid learning rate provided. Aborting..."<<endl;
            return;
        }
        if(properties.isSet(Property::hyperParameters::momentumConstant))
            momentumConstant = stod(properties.getProperty(Property::hyperParameters::momentumConstant));
        if(properties.isSet(Property::hyperParameters::iterCount))
            IterCnt = stoi(properties.getProperty(Property::hyperParameters::iterCount));
        else{
            cout<<"IterCount either set to 0 or not provided. Exiting...";
            return;
        }
        if(properties.isSet(Property::hyperParameters::batchSize))
            batch_size = stoi(properties.getProperty(Property::hyperParameters::batchSize));


        cout<<"\n\tStarting batch training.\n\n";

        cout<<"Prediction Accuracy before training: "<<accuracy(m_Inputs, m_Lables)<<endl<<endl;
        double acc;
        vector<mat> Theta_grad_prev(m_Theta.capacity());     //To handle momentum
        for(int k = 0;k<Total/batch_size; ++k){
            mat Input_batch = m_Inputs.rows(batch_size*(k),batch_size*(k+1)-1);
            mat Label_batch = m_Lables.rows(batch_size*(k),batch_size*(k+1)-1);

            for(unsigned i=0;i<m_Theta.capacity();++i)
                Theta_grad_prev[i] = zeros<mat>(size(m_Theta[i]));


            cout<<"Batch "<<k+1<<endl;
            for(int i=0; i<IterCnt; ++i){
                cout<<"\tIteration "<<i+1<<endl;

                vector<mat> Theta_grad = backpropogate(Input_batch, Label_batch, regularizationParameter);

                for(unsigned i=0;i<m_Theta.capacity();++i){
                    Theta_grad[i] += momentumConstant*Theta_grad_prev[i];
                    m_Theta[i] -= learningRate*Theta_grad[i];
                    Theta_grad_prev[i] = Theta_grad[i];
                }
            }
        }
        acc = accuracy(m_Inputs, m_Lables);
        cout<<"Prediction Accuracy on training set: "<<acc<<endl;
        if(m_inpPaths.size()-1){
            if(load(m_inpPaths.at(m_inpPaths.size()-1))==true){
                cout<<"\n\nUsing test set"<<endl;
                cout<<"Prediction Accuracy on test set: "<<accuracy(m_Inputs, m_Lables)<<endl;
            }
            else
                cout<<"Could not load test set. Training may be incomplete."<<endl;
        }
        else
            cout<<"No test set provided."<<endl;
        save(m_paramPath);
    }

    //TODO: remove need for this.
    //Possible fix: return a struct with all sorts of training stats. Handle this outside.
    void network::train(string input, int label){//regularization parameter and learning rate and a momentum constant
        double regularizationParameter = 0, learningRate = 0, momentumConstant = 0;

        //Get hyperParameters from property tree
        if(properties.isSet(Property::hyperParameters::regularizationParameter))
            regularizationParameter = stod(properties.getProperty(Property::hyperParameters::regularizationParameter));
        if(properties.isSet(Property::hyperParameters::learningRate))
            learningRate = stod(properties.getProperty(Property::hyperParameters::learningRate));
        else{
            cout<<"No valid learning rate provided. Aborting..."<<endl;
            return;
        }
        if(properties.isSet(Property::hyperParameters::momentumConstant))
            momentumConstant = stod(properties.getProperty(Property::hyperParameters::momentumConstant));

        cout<<"\n\tStarting individual training.\n\n";

        mat Input = zeros(28,28);
        mat Label = zeros(1,1);
        Input.load(input);
        Input.reshape(1,784);
        Label.at(0) = label;
        cout<<"Overall Prediction Accuracy before training: "<<accuracy(Input, Label)<<endl<<endl;
        vector<mat> Theta_grad_prev(m_Theta.capacity());     //To handle momentum
        for(unsigned i=0;i<m_Theta.capacity();++i)
            Theta_grad_prev[i] = zeros<mat>(size(m_Theta[i]));
        int i=0;
        while(1){
            cout<<"\tIteration "<<++i<<endl;
            vector<mat> Theta_grad = backpropogate(Input, Label, regularizationParameter);
            for(unsigned i=0;i<m_Theta.capacity();++i){
                Theta_grad[i] += momentumConstant*Theta_grad_prev[i];
                m_Theta[i] -= learningRate*Theta_grad[i];
                Theta_grad_prev[i] = Theta_grad[i];
            }
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

    class ConvNet: public network{
            int m_imgHt,m_imgWth, m_strideLn;
            vector<int> m_convSizes;
            vector<mat> m_kernels;
        public:
            ConvNet(propertyTree);
            mat backpropogate(mat, mat, double);
            mat predict(mat);
            mat output(mat);
            vector<mat> forwardPass(cube);
    };

    ConvNet::ConvNet(propertyTree properties):network(properties){
        this->properties = properties;
        m_imgHt = 28;
        m_imgWth = 28;
        m_strideLn = 1;
//        m_convSizes = convSizes;
//        for(unsigned int i=0;i<m_convSizes.size();++i){
          for(int j=0;j<4;++j)
            m_kernels.push_back(randWeights(3,3));
//        }
    }

    cube conv(cube img, vector<mat> kernels){
        cube output  = zeros<cube>(img.n_rows,img.n_cols,kernels.size()*img.n_slices);
        for(unsigned int i=0;i<img.n_slices;++i)
            for(unsigned int j=0;j<output.n_slices;++j)
                output.slice(j+i)= conv2(img.slice(i),kernels[i],"same");
        return output;
    }

    cube pool(cube layer, int downSamplingFactor){
        cube output = zeros<cube>(layer.n_rows/downSamplingFactor,layer.n_cols/downSamplingFactor,layer.n_slices);
        for(unsigned int i=0;i<layer.n_slices;++i){
            for(unsigned int j=0;(j+1)*downSamplingFactor<output.n_rows;++j)
                for(unsigned int k=0;(k+1)*downSamplingFactor<output.n_cols;++k)
                    output.slice(i).at(j,k) = accu(layer.slice(i).submat(j*downSamplingFactor,k*downSamplingFactor,
                                                   (j+1)*downSamplingFactor,(k+1)*downSamplingFactor));
        }
        return output;
    }

    vector<mat> ConvNet::forwardPass(cube img){
        cout<<"running ConvNet::forwardPass"<<endl;
        cube convLayer = conv(img,m_kernels);
        cube activated = zeros<cube>(size(convLayer));
        for(unsigned int i=0;i<convLayer.n_slices;++i)
            activated.slice(i) = RectifiedLinearUnitActivation(convLayer.slice(i));
        cube pooledLayer = pool(activated,2);
        mat input = vectorise(pooledLayer).t();
        return network::forwardPass(input);
    }
    mat ConvNet::predict(mat Input){
        Input.reshape(28,28);
        cube input(Input.n_rows,Input.n_cols,1);
        input.slice(0) = Input;
        vector<mat> layers = forwardPass(input);
        int InputSize = input.n_slices;
        mat pred = zeros(InputSize,1);
        for(int i=0; i<InputSize; ++i){
            pred(i) = layers.back().row(i).index_max();
        }
        return pred;
    }
    mat ConvNet::output(mat Input){
        cout<<"running ConvNet::output"<<endl;
        int InputSize = Input.n_rows;
        Input.reshape(28,28);
        cube input(Input.n_rows,Input.n_cols,1);
        input.slice(0) = Input;
        vector<mat> layers = forwardPass(input);
        cout<<"Output: "<<layers.back()<<endl;
        cout<<"Sum of outputs"<<accu(layers.back())<<endl;
        mat pred = zeros(InputSize,1);
        for(int i=0; i<InputSize; ++i){
            pred(i) = layers.back().row(i).max();
        }
        return pred;
    }
}
#endif // INTELLIDETECT_H_INCLUDED
