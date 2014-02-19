//Matlab interface

#include "mex.h"
#include <vector>
#include <time.h>
#include "dopshift.hpp"
#include <iostream>

mxArray * toMAT(const Eigen::MatrixXd& x)
{
  mxArray * r = mxCreateNumericMatrix(x.rows(), x.cols(), mxDOUBLE_CLASS, mxREAL);
  if(!r)
        return r;
  int n =  (x.rows()*x.cols());
  double * p = mxGetPr(r);
  
  for ( int index = 0; index < n ; index++ ) 
    p[index] = x(index);
  return r;
}

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]){

  mwSize xmrows,xncols, layermrows, layerncols, rmrows, rncols;
  std::vector<int> layerv;
  double *layervtemp;
  double *xv;
  double *rv;
  int nriter = 200;
  double omitcue = 0;
  double late = 0;
  
  if(nrhs != 2)
    mexErrMsgIdAndTxt( "MATLAB:xtimesy:invalidNumInputs",
            "Two inputs required.");
  if(nlhs!=2) 
    mexErrMsgIdAndTxt( "MATLAB:xtimesy:invalidNumOutputs",
            "Two outputs required.");

  omitcue = mxGetScalar(prhs[0]); 
  late = mxGetScalar(prhs[1]);

  std::cout << late;
  
  Output output;
  Dopshift(output, bool(omitcue), bool(late));

  plhs[0] = toMAT(output.episodes); //episodes
  plhs[1] = toMAT(output.rest); //rest 
}
