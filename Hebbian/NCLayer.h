#pragma once
#include "NCNeuron.h"

typedef std::shared_ptr<NCNeuron> NCNeuronPtr;

class NCLayer{
    
private:
     
     std::vector<NCNeuronPtr> neurons;

     
public:
    
void setparameters(){
}


NCLayer(unsigned layersize,std::vector<NCNeuronPtr> inputneurons){
    setparameters();
    neurons.reserve(layersize);
    std::vector<NCNeuronPtr> neuronstemp;

    for(unsigned i = 0;i<layersize;i++){
        NCNeuronPtr neuron0ptr(new NCNeuron(inputneurons));
        neurons.push_back(neuron0ptr);
	}

}

NCLayer(unsigned layersize){ //Constructor for input layer
    setparameters();
    neurons.reserve(layersize);
    for(unsigned i = 0;i<layersize;i++){
        NCNeuronPtr inputtemp(new NCNeuron());
        neurons.push_back(inputtemp);
    }
        
}

NCLayer(std::vector<NCNeuronPtr> inputneurons){ //Constructor for output layer
    setparameters();
    neurons.reserve(1);
    NCNeuronPtr neuron0ptr(new NCNeuron(inputneurons));
    neurons.push_back(neuron0ptr);
    neurons[0]->outputinit();
}


void run(){
   for(unsigned i = 0; i < neurons.size(); i++){
        neurons[i]->run();
    }
}



void adapt(const double& rsignal){
    double residualerror = rsignal;
    for(unsigned i = 0; i < neurons.size(); i++){ 
      neurons[i]->adapt(residualerror);
        residualerror = residualerror - *neurons[i]->gety(); 
       
    }
}

void reset(){
    for(unsigned i = 0; i < neurons.size(); i++){
        neurons[i]->reset();
    }
}

void addgrad(const double& rsignal){
    double residualerror = rsignal;
    for(unsigned i = 0; i < neurons.size(); i++){
        neurons[i]->addgrad(rsignal);
        residualerror = residualerror - *neurons[i]->gety(); 
        
    }
}

void episodicadapt(){
    for(unsigned i = 0; i < neurons.size(); i++){

        neurons[i]->episodicadapt();
    }
}

void resetelig(){
    for(unsigned i = 0; i < neurons.size(); i++){
        neurons[i]->resetelig();
    }
}

void outputinit(){
    for(unsigned i = 0; i < neurons.size(); i++){
        neurons[i]->outputinit();
    }
}

Eigen::MatrixXd getw(){
    if(neurons.size() > 0){
    Eigen::VectorXd wfirst = neurons[0]->getw();
    Eigen::MatrixXd W(neurons.size(), wfirst.size());
    W.row(0) = wfirst;
    for(unsigned i = 1;i<neurons.size();i++)
            W.row(i) = neurons[i]->getw();
    return(W);
     }

}

std::vector<NCNeuronPtr> getconnections(){
        return(neurons);
}

Eigen::VectorXd output(){
    if(neurons.size() > 0){
        Eigen::VectorXd y(neurons.size());
        for(unsigned i = 0;i<neurons.size();i++)
            y(i) = *neurons[i]->gety(); 
    return(y);
    }
    else{
        Eigen::VectorXd yr(1,1);
        yr(0) = 0;
        return(yr);
    }
}

void sety(const Eigen::VectorXd& x){
    if (x.rows() == int(neurons.size())){
        for(unsigned i = 0;i<neurons.size();i++)
            neurons[i]->sety(x(i));
    }
}
    
};
