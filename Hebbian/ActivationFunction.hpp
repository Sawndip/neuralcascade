#pragma once
#include <functional>

double dtanheff(double x)  __attribute__ ((visibility ("internal")));
double tanheff(double x)   __attribute__ ((visibility ("internal")));
double dlinact(double x)  __attribute__ ((visibility ("internal")));
double linact(double x)  __attribute__ ((visibility ("internal")));


typedef std::shared_ptr<std::pointer_to_unary_function<double, double> > functorptr;

inline double dradbas(double x){

    
     if( x < 0.0)
         return 0.0;
     else
         return 1.0;
}

inline double radbas(double x){
    //Efficient tanh approximation using 
         return (exp(-x*x));
}

inline double dpartlin(double x){

    if( x < 0.0 )
        return 0.01;
    else
        return 1;
        
         
}

inline double partlin(double x){
    //Efficient tanh approximation using 
     if( x < 0.0 )
         return x/100.0;
     else
         return x;
}




inline double dtanheff(double x){

     if( x < -3.0 )
         return 0.0;
     else if( x > 3.0 )
         return 0.0;
     else{
        double dtemp = 1.2*((x * x) -  9.0) / (3.0 * (( x * x)  + 3.0));
        return (dtemp*dtemp);   
     }
}

inline double tanheff(double x){
    
    //Efficient tanh approximation using 
     if( x < -3.0 )
         return -1.2;
     else if( x > 3.0 )
         return 1.2;
     else
        return 1.2 * x * ( 27.0 + (x * x) ) / ( 27.0 + (9.0 * x * x) );
}


inline double dctanheff(double x){

     if( x < -2.501 )
         return -0.1;
     else if( x > 2.501 )
         return -0.1;
     else{
        double dtemp = 1.2*((x * x) -  9.0) / (3.0 * (( x * x)  + 3.0));
        return (dtemp*dtemp);   
     }
}

inline double ctanheff(double x){
    
    //Efficient tanh approximation using 
     if( x < -2.501 )
         return -1.1982;
     else if( x > 2.501 )
         return 1.1982;
     else
        return 1.2 * x * ( 27.0 + (x * x) ) / ( 27.0 + (9.0 * x * x) );
}
// inline double dptanheff(double x){
// 
//     
//      if( x < -2.501 )
//          return 0.01173;
//      else if( x > 2.501 )
//          return 0.01173;
//      else{
//         double dtemp = 0.6*((x * x) -  9) / (3 * (( x * x)  + 3));
//         return (dtemp*dtemp);   
//      }
// }
// 
// inline double ptanheff(double x){
//     //Efficient tanh approximation using 
//      if( x < -2.995 )
//          return 0;
//      else if( x > 2.995 )
//          return 1.198;
//      else
//         return (0.6 + 0.6 * x * ( 27 + (x * x) ) / ( 27 + (9 * x * x) ));
// }
// 

inline double dlinact(double x){

return double(1);
}

inline double linact(double x){
        return x;
}




class ActivationFunction{
    private: 
        functorptr  func;
        functorptr  dfunc;

    public:
        ActivationFunction( double  (*funct)(double), double  (*dfunct)(double) ){
        func.reset(new std::pointer_to_unary_function<double, double>(*funct));
        dfunc.reset(new std::pointer_to_unary_function<double, double>(*dfunct));
        }
        
        ActivationFunction(){
        func.reset(new std::pointer_to_unary_function<double, double>(&tanheff));
        dfunc.reset(new std::pointer_to_unary_function<double, double>(&dtanheff));
        }
        
//         double operator()(double x){
//             return (func(x));
//         }
        
        std::pointer_to_unary_function<double, double> dydx(){
            return (*dfunc);
        }
        
        std::pointer_to_unary_function<double, double> y(){
            return (*func);
        }
};

