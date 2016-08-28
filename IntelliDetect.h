#ifndef INTELLIDETECT_H_INCLUDED
#define INTELLIDETECT_H_INCLUDED
#include <armadillo>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <ctime>
#define INTELLI_VERSION "2.2"
/* This header file contains definitions for functions for handling various ANN processes */

using namespace arma;
using namespace std;

namespace IntelliDetect{

    struct propertyTree{
            string data;
            vector< pair<string, propertyTree> > childNodes;
        public:
            //TODO:
            //propertyTree(string);
            string getProperty(string);
            void setProperty(string,string);
            bool isSet(string);
            string toString(int);
    };

    void propertyTree::setProperty(string property, string value){
        bool flag=false;
        string nodeName(""),propName("");
        for(unsigned int i=0;i<property.length();++i){
            if(property[i]=='.'){
                flag=true;
                ++i;
            }
            if(flag)
                propName.append(&property[i],1);
            else
                nodeName.append(&property[i],1);
        }
        for(int i=0;i<childNodes.size();++i){
            if(!get<0>(childNodes[i]).compare(nodeName)){
                get<1>(childNodes[i]).setProperty(propName,value);
                return;
            }
        }
        childNodes.push_back(make_pair(nodeName,*(new propertyTree)));
        if(propName.length()){
            get<1>(childNodes.back()).setProperty(propName,value);
            return;
        }
        else
            get<1>(childNodes.back()).data = value;
        cout<<"Property set"<<endl;
    }

    string propertyTree::getProperty(string property){
        bool flag=false;
        string nodeName,propName;
        for(unsigned int i=0;i<property.length();++i){
            if(property[i]=='.'){
                flag=true;
                ++i;
            }
            if(flag)
                propName.append(&property[i],1);
            else
                nodeName.append(&property[i],1);
        }
        for(auto prop: childNodes){
            if(!get<0>(prop).compare(nodeName)){
                return get<1>(prop).getProperty(propName);

            }
        }
        return data;
    }

    bool propertyTree::isSet(string property){
        bool flag=false;
        string nodeName,propName;
        for(unsigned int i=0;i<property.length();++i){
            if(property[i]=='.'){
                flag=true;
                ++i;
            }
            if(flag)
                propName.append(&property[i],1);
            else
                nodeName.append(&property[i],1);
        }
        for(auto prop: childNodes){
            if(!get<0>(prop).compare(nodeName)){
                if(propName.length())
                    return get<1>(prop).isSet(propName);
                else
                    return true;

            }
        }
        return false;
    }

    string propertyTree::toString(int level=0){
        string strRep("");
        if(!level){
            strRep+=string("version = ");
        }
        strRep+=(data+string("\n"));
        for(auto prop: childNodes){
            for(int i=0;i<level;++i)
                strRep+=string("\t");
            strRep+=(get<0>(prop) + string(" = ") + get<1>(prop).toString(level+1));
        }
        return strRep;
    }
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

    class network{
            mat m_X,m_Y;
            mat m_Theta1, m_Theta2, m_nn_params;
            int m_inputLayerSize,m_hiddenLayerSize,m_outputLayerSize;
            string m_paramPath;
            vector<string> m_inpPaths;
            mat (*activation)(mat);
            mat (*activationGradient)(mat);
            propertyTree properties;

        public:

            vector<long double> trainSetCostsReg;
            vector<long double> trainSetCosts;
            network(mat (*activationPtr)(mat), mat (*activationGradientPtr)(mat),string, int, int, int);
            mat randInitWeights(int, int);
            mat predict(mat);
            mat predict(string);
            mat output(string);
            mat output(mat);
            mat backpropogate(mat, mat, double);
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

    network::network(mat (*activationPtr)(mat) = sigmoid, mat (*activationGradientPtr)(mat) = sigmoidGradient, string param = "", int inpSize = 784, int HdSize = 100, int OpSize = 10){
        properties.data = string("Version ")+string(INTELLI_VERSION);
        properties.setProperty("Id","TestNetwork");
        properties.setProperty("Network.inputLayerSize",to_string(inpSize));
        properties.setProperty("Network.hiddenLayerSize",to_string(HdSize));
        properties.setProperty("Network.outputLayerSize",to_string(OpSize));

        m_inputLayerSize = inpSize;
        m_hiddenLayerSize = HdSize;
        m_outputLayerSize = OpSize;
        m_paramPath = param;
        activation = activationPtr;
        activationGradient = activationGradientPtr;
        if(m_nn_params.load(param)==false){
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
        trainSetCosts.resize(0);
        trainSetCostsReg.resize(0);
    }

    bool network::load(vector<string> paths, int startInd=0, int endInd=0){
        if(!paths.size()){
            cout<<"network::load(): error: Input file list is empty.";
            return false;
        }

        m_inpPaths = paths;
        mat tmpX;
        m_X.resize(0,0);
        unsigned int i=0;
        do{
            tmpX.load(m_inpPaths.at(i++));
            m_X = join_vert(m_X,tmpX);
        }while(i<m_inpPaths.size()-1); //last path represents test set.)
        m_X = shuffle(m_X);
        if(endInd)
            m_X = m_X.rows(startInd,endInd);
        else
            m_X = m_X.rows(startInd,m_X.n_rows-1);
        m_Y = m_X.col(0);
        m_X = m_X.cols(1,m_X.n_cols-1);

        return true;
    }

    bool network::load(string path, int startInd=0, int endInd=0){
        if(m_X.load(path)==false)
            return false;
        m_X = shuffle(m_X);
        if(endInd)
            m_X = m_X.rows(startInd,endInd);
        else
            m_X = m_X.rows(startInd,m_X.n_rows-1);
        m_Y = m_X.col(0);
        m_X = m_X.cols(1,m_X.n_cols-1);

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
        m_X = Inputs;
        m_Y = Labels;

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
        if(properties.isSet("Id"))
            folderName = string(properties.getProperty("Id"));
        else{
            time(&Time);
            folderName = string("network")+string(asctime(localtime(&Time))+string("/"));
        }
        fullpath += folderName;
        fstream trainStat;
        //Save training stats
        trainStat.open(fullpath+string("trainingStats.csv"),ios::out);
        for(int i=0;i<trainSetCosts.size();++i){
            trainStat<<trainSetCosts[i]<<", "<<trainSetCostsReg[i]<<", "<<endl;
        }
        trainStat.close();
        //Save weights
        mat hyper_params = {
                            static_cast<double>(m_inputLayerSize),
                            static_cast<double>(m_hiddenLayerSize),
                            static_cast<double>(m_outputLayerSize)
                           };
        mat tmp_params = join_vert(vectorise(hyper_params),m_nn_params);
        tmp_params.save((fullpath+string("parameters.csv")),csv_ascii);
        //Save propertyTree
        fstream configFile;
        configFile.open(fullpath+string("network.conf"),ios::out);
        configFile<<string("//Intellidetect Configuration file\n");
        configFile<<properties.toString();
        configFile.close();
        return true;
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
        mat h2 = sigmoid(h1*trans(m_Theta2));
        mat pred = zeros(InputSize,1);
        for(int i=0; i<InputSize; ++i){
            pred(i) = h2.row(i).index_max();
        }
        return pred;
    }
    mat network::output(mat Input){
        int InputSize = Input.n_rows;
        Input = join_horiz(ones<mat>(InputSize,1),Input);
        mat z2 = Input*trans(m_Theta1);
        mat h1 = join_horiz(ones<mat>(z2.n_rows,1),activation(z2));
        mat h2 = sigmoid(h1*trans(m_Theta2));
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
        inputMat.reshape(1,784);
        return predict(inputMat);
    }

    mat network::output(string input){
        mat inputMat;
        inputMat.load(input);
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
            mat a3 = sigmoid(z3);
            output_tmp(as_scalar(Outputs(i))) = 1;

            cost += as_scalar(accu(output_tmp%log(a3)+(1-output_tmp)%log(1-a3)))/InputSize*(-1);

            mat delta_3 = a3-output_tmp;
            mat delta_2 = trans(Theta2.cols(1,Theta2.n_cols-1))*delta_3%activationGradient(z2);
            mat delta_2_check = trans(Theta2.cols(1,Theta2.n_cols-1))*delta_3%numericalGradient(z2);
            cout<<"Diff in activation grad and num_grad: "<<accu(delta_2-delta_2_check)<<endl;
            Theta1_grad += delta_2*CurrentInput.t();
            Theta2_grad += delta_3*a2.t();
            output_tmp(as_scalar(Outputs(i)),0) = 0;
        }
        cout<<"\tCost(unregularized) = "<<cost;
        trainSetCosts.push_back(cost);
        cost += (accu(square(Theta1.cols(1,Theta1.n_cols-1)))+accu(square(Theta2.cols(1,m_hiddenLayerSize))))*lambda/(2*InputSize);
        cout<<"\t\tCost (regularized) = "<<cost<<endl;
        trainSetCostsReg.push_back(cost);
        Theta1_grad /= InputSize;
        Theta2_grad /= InputSize;

        Theta1_grad += join_horiz(zeros<mat>(Theta1.n_rows,1), (lambda/InputSize)*Theta1.cols(1,Theta1.n_cols-1));
        Theta2_grad += join_horiz(zeros<mat>(Theta2.n_rows,1), (lambda/InputSize)*Theta2.cols(1,Theta2.n_cols-1));

        mat grad = join_vert(vectorise(Theta1_grad),vectorise(Theta2_grad));
        return grad;
    }

    void network::train(double lambda = 0.5,double alpha = 0.05,double mu = 1){//regularization parameter and learning rate and a momentum constant
        int Total = m_X.n_rows, batch_size = m_X.n_rows/100, IterCnt = 35;
        if(!batch_size){
            batch_size = 1;
            IterCnt = 10;
        }
        cout<<"\n\tStarting batch training.\n\n";

        cout<<"Prediction Accuracy before training: "<<accuracy(m_X, m_Y)<<endl<<endl;

        for(int k = 0;k<Total/batch_size; ++k){
            mat X_batch = m_X.rows(batch_size*(k),batch_size*(k+1)-1);
            mat Y_batch = m_Y.rows(batch_size*(k),batch_size*(k+1)-1);
            cout<<"Batch "<<k+1<<endl;
            for(int i=0; i<IterCnt; ++i){
                cout<<"\tIteration "<<i+1<<endl;
                m_nn_params = mu*m_nn_params - alpha*backpropogate(X_batch, Y_batch, lambda);
            }
        }
        m_Theta1 = reshape(m_nn_params.rows(0,(m_inputLayerSize+1)*(m_hiddenLayerSize)-1),m_hiddenLayerSize,m_inputLayerSize+1);
        m_Theta2 = reshape(m_nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSize),m_nn_params.size()-1), m_outputLayerSize, m_hiddenLayerSize+1);

        cout<<"Prediction Accuracy on training set: "<<accuracy(m_X, m_Y)<<endl;

        if(m_inpPaths.size()){
            if(load(m_inpPaths.at(m_inpPaths.size()-1))==true){
                cout<<"\n\nUsing test set"<<endl;
                cout<<"Prediction Accuracy on test set: "<<accuracy(m_X, m_Y)<<endl;
            }
            else
                cout<<"Could not load test set. Training may be incomplete."<<endl;
        }
        else
            cout<<"No training set provided."<<endl;
        save(m_paramPath);
    }

    void network::train(string input, int label, double lambda = 0.5,double alpha = 0.05,double mu = 1){//regularization parameter and learning rate and a momentum constant


        cout<<"\n\tStarting individual training.\n\n";

        mat X = zeros(28,28);
        mat Y = zeros(1,1);
        X.load(input);
        X.reshape(1,784);
        Y.at(0) = label;
        cout<<"Overall Prediction Accuracy before training: "<<accuracy(X, Y)<<endl<<endl;

        int i=0;
        while(1){
            cout<<"\tIteration "<<++i<<endl;
            m_nn_params = mu*m_nn_params - alpha*backpropogate(X, Y, lambda);
            m_Theta1 = reshape(m_nn_params.rows(0,(m_inputLayerSize+1)*(m_hiddenLayerSize)-1),m_hiddenLayerSize,m_inputLayerSize+1);
            m_Theta2 = reshape(m_nn_params.rows((m_inputLayerSize+1)*(m_hiddenLayerSize),m_nn_params.size()-1), m_outputLayerSize, m_hiddenLayerSize+1);
            mat out = predict(X);
            mat conf = output(X);
            cout<<"out.at(0) = "<<out.at(0)<<endl<<"confidence = "<<conf(0)<<endl;
            if(out.at(0) == label && conf(0)>0.5)
                break;
        }

        cout<<"Done training on given example."<<endl;
        cout<<"Overall Prediction Accuracy after training: "<<accuracy(X, Y)<<endl<<endl;

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
            mat m_X,m_Y;
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
