#pragma once
#include "NCLayer.h"

typedef std::shared_ptr<NCLayer> NCLayerPtr;

class NCNet{
private:
     std::vector<NCLayerPtr> layers;

public:
NCNet(Eigen::VectorXi layersizes)
{
    layers.reserve(layersizes.size()+1);
    NCLayerPtr layertemp(new NCLayer(layersizes(0)));
    layers.push_back(layertemp);
    for(unsigned i = 1;i<layersizes.size();i++)
    {   
        NCLayerPtr layertemp(new NCLayer(layersizes(i), layers[i-1]->getconnections()));
        layers.push_back(layertemp);
    }
    NCLayerPtr layertemp2(new NCLayer(layers[layersizes.size()-1]->getconnections()));
    layers.push_back(layertemp2);
    reset();
}

Eigen::VectorXd output(){
    if(layers.size()>0)
      return(layers[(layers.size()-1)]->output());
    else{
      		return(Eigen::VectorXd::Zero(1));
	}
}

void run(const Eigen::VectorXd& input){
    layers[0]->sety(input);
    for(unsigned i = 1;  i < layers.size();i++){
        layers[i]->run();
    }
}

void adapt(const double& rsignal){
    for(unsigned i = 1; i < layers.size(); i++){
        layers[i]->adapt((rsignal));
    }
}

void reset(){
    for(unsigned i = 0; i < layers.size(); i++){
        layers[i]->reset();
    }
    layers[layers.size()-1]->outputinit();
}

void resetelig(){
    for(unsigned i = 1; i < layers.size(); i++){
        layers[i]->resetelig();
    }
}

void addgrad(const double& rsignal){
    for(unsigned i = 1; i < layers.size(); i++){
        layers[i]->addgrad(rsignal);
    }
}

void episodicadapt(){
    for(unsigned i = 1; i < layers.size(); i++){
        layers[i]->episodicadapt();
    }
}

Eigen::MatrixXd getw(){
    return(layers[layers.size()-1]->getw());
}

};
