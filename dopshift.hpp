
#pragma once
#include "Hebbian/NCNet.h"
#include <deque>
#include <memory>
#include <random>


typedef std::shared_ptr<NCNet> CNetPtr;

struct Output{ 
    Eigen::VectorXd episodes;
    Eigen::VectorXd steps;
    Eigen::MatrixXd rest;
    Eigen::MatrixXd weights;
};

int Dopshift(Output & output, bool omitcue, bool late);

int Dopshift(Output & output, bool omitcue, bool late){
    
    CNetPtr BotCritic;
	int ntrials = 100;
    int nnets = 100;
    int selectdiagidx = 0;
	int nodim = 4;
	int nadim = 1; 
    int distcounter = 0;
	double rsignal = 0.0;
	double r_est = 0.0;
	double r_est_prev = 0.0;
    int oddball = 0;
    double rewardavg = 0.0;
	Eigen::VectorXi clayersizes;
	clayersizes.resize(2);
	clayersizes << (nodim), 7;

	Eigen::VectorXd _criticInput;
	Eigen::VectorXd _actorInput;
	Eigen::VectorXd _actorOutput;
	Eigen::VectorXd avgrsignal;
	Eigen::VectorXd countrsignal;
    Eigen::VectorXd avgrest;
    
	avgrsignal.resize(40);
	avgrsignal = Eigen::VectorXd::Zero(40);
	countrsignal.resize(40);
	countrsignal = Eigen::VectorXd::Zero(40);
    avgrest.resize(40);
	avgrest = Eigen::VectorXd::Zero(40);
	_actorInput.resize(nodim);
	_criticInput.resize(nadim);


	double* observation = new double[nodim];
	double* input = new double[nodim];

	std::random_device rd;
  	std::mt19937 gen(rd());
   	std::uniform_real_distribution<> dis(0, 1);

    
    if(late == 1)
        ntrials = 400;
    
    for(int netidx = 0; netidx < nnets;netidx++){
        double reward = 0;
        BotCritic = std::make_shared<NCNet>(clayersizes);
        _criticInput = Eigen::VectorXd::Zero(nodim);
        _actorInput = Eigen::VectorXd::Zero(nodim);
        observation[0] = 0;
        observation[1] = 0;
        observation[2] = 0;
        observation[3] = 0;
        int cuecounter = -1;
        BotCritic->reset();
        
    int trialidx = 0;
	while(trialidx < ntrials){
		reward = 0;
		if(cuecounter < 0){
            if(distcounter > 100){
				cuecounter = 39;
                distcounter = 0;
			}
            else
                distcounter ++;
		}
		else{
			cuecounter--;
			if(cuecounter == 21)
				observation[0] = observation[0] + 1.0;
            
			if(cuecounter == 1){
                if((oddball == 0) or (oddball == 1)){
                    reward = reward + 1.0;
                    
                }
                oddball = 0;
                observation[0] = 0;
                observation[1] = 0;
                observation[2] = 0;
                observation[3] = 0;
			}
            if(cuecounter == 2 ){
                if(oddball == 1){
                }
                else{
                    observation[3] = observation[3] + 1.0;
                    oddball = 0;
                }
                
			}
			if(cuecounter == 6){           
                if((trialidx+1) == ntrials){
                        oddball = 0;
                        if(omitcue == 0)
                            observation[1] = observation[1] + 1.0;
                }
                else{
                    if(dis(gen) < 0.4){
                        if(dis(gen) < 0.5){
                            oddball = 2;
                            observation[2] = observation[2] + 1.0;
                        }
                        else{
                    	oddball = 1;
                        }
                    }
                    else{
                        observation[1] = observation[1] + 1.0;
                        oddball = 0;
                    }
                }
			}

		}
		for(int i = 0;i < nodim;i++){
			_actorInput(i) = observation[i];
		}
        
		//Handle networks and internal signals
        BotCritic->adapt(reward); 
		_criticInput << _actorInput;
        BotCritic->run(_criticInput);
        
	 	r_est_prev = r_est;
   	 	r_est = (BotCritic->output())(0);
		rsignal = 0.02*reward + 0.98*r_est - r_est_prev;
        
		if( (cuecounter >= 0) && ((trialidx + 1) == ntrials)){
			avgrsignal[39 - cuecounter] = avgrsignal[39 - cuecounter] + rsignal;
            avgrest[39 - cuecounter] = avgrest[39 - cuecounter] + r_est;
			countrsignal[39 - cuecounter]++;
            
        }
		if(cuecounter == 0){
			cuecounter = -1;
          trialidx++;
            }
		rewardavg = 0.99998*rewardavg + 0.00002*reward;//EWMA reward for faster learning
           }
	}
	output.episodes = avgrsignal;
	output.rest = (avgrest.array() / countrsignal.array()).matrix();
	
}
