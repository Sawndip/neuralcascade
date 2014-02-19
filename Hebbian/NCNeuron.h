#pragma once
#include <memory>
#include <random>
#include "Eigen/Core"
#include "ActivationFunction.hpp"
    
class NCNeuron{

//Includes unused functionality such as episodic weight updates
   
private:
    typedef std::shared_ptr<NCNeuron> NCNeuronPtr;
    double supstep;
    double b;
    bool episodic = false;
    double y;
    double wx;
    double weligdecay;
    double weightdecay;
    double r;
    std::shared_ptr<ActivationFunction> actfunc;
    std::vector<NCNeuronPtr> inputconnections;
    Eigen::VectorXd w;
    Eigen::VectorXd welig;
    Eigen::VectorXd weligd;
    Eigen::VectorXd dw;
    Eigen::VectorXd x;
    
public:

void setparams(){
    supstep = 0.000003;
    weligdecay = 0.02; 
    weightdecay = 0.0;
    actfunc.reset(new ActivationFunction);
}

void outputinit(){
    reset();
    supstep = 0.001;
    w = Eigen::VectorXd::Zero(w.size());
    w(0) = 1;
    weightdecay = 0.0;
    b = 0.0;
    actfunc.reset(new ActivationFunction(&linact, &dlinact));
	
}

NCNeuron(std::vector<NCNeuronPtr> inputs){
    this->addInputs(inputs);
    setparams();
    reset();
}

NCNeuron(){
    setparams();
    reset();    
}

void setactivationfunc(){

}

void addgrad(const double& rsignal){
  dw = dw.eval() - (welig.eval() - (rsignal*weligd)).eval();
}

void adapt(const double& rsignal){
 w = (1.0 - (supstep * weightdecay) ) * w.eval() - supstep*(welig.eval() - (rsignal*weligd)).eval();
}

void episodicadapt(){
  w = w + supstep*dw;
  dw = 0.0*dw;
}

void run(){
     if(inputconnections.size() > 0)
     {   
      for(unsigned i = 0;i<inputconnections.size();i++){
        x(i) = *(inputconnections[i]->gety());
        
      }
      x(inputconnections.size()) = b;
        wx = (x.transpose()*w).eval()(0);
        y = actfunc->y()(wx); 
    welig = (welig * (double(1) - weligdecay)).eval() + (((actfunc->dydx()(wx) * y)) * x); //rightside weligdecay just for more elegant values with different windows, approx. equivalent to moving running average
	weligd = (weligd * (double(1) - weligdecay)).eval() + ((actfunc->dydx()(wx)) * x); //as above
}
else{

}
}

void reset(){
     y = 0.0;
     wx = 0.0;
     x = Eigen::VectorXd::Zero(inputconnections.size()+1,1);
     b = 0.0001;
    if(inputconnections.size()>0){
        w = 0.003*Eigen::VectorXd::Random(inputconnections.size()+1);
	welig = 0.0*Eigen::VectorXd::Random(inputconnections.size()+1);
	weligd = 0.0*Eigen::VectorXd::Random(inputconnections.size()+1);
}
    else{
        w = Eigen::VectorXd::Constant(1,1,1);
	welig = Eigen::VectorXd::Constant(1,1,1);
	weligd = Eigen::VectorXd::Constant(1,1,1);
}
     if(episodic)
	dw = 0.01*Eigen::VectorXd::Zero(inputconnections.size()+1);
}

void resetelig(){
    if(inputconnections.size()>0){
	welig = 0.0*Eigen::VectorXd::Random(inputconnections.size()+1);
	weligd = 0.0*Eigen::VectorXd::Random(inputconnections.size()+1);
}
    else{
	welig = Eigen::VectorXd::Constant(1,1,1);
	weligd = Eigen::VectorXd::Constant(1,1,1);
}
     if(episodic)
	dw = 0.01*Eigen::VectorXd::Zero(inputconnections.size()+1);
}

void sety(const double& yin){
    y = yin;
}

double* gety(){
    return(&y);
}

Eigen::VectorXd getw(){
    return(w);
}

void addInputs(std::vector<NCNeuronPtr> newinputs){ 
    inputconnections.reserve(newinputs.size() + inputconnections.size());
	for(unsigned i = 0; i < newinputs.size(); i++){
        inputconnections.push_back(newinputs[i]);
    }
}


};
