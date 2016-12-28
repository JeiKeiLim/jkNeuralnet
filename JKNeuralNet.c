//
//  JKNeuralNet.c
//  NeuralNet
//
//  Created by JeiKei on 2016. 12. 6..
//  Copyright © 2016년 JeiKei. All rights reserved.
//

#include "JKNeuralNet.h"

void jknn_train(jknn_param *param, double** x, char** y, int length)
{
    double h[length][param->n_hidden];      // hidden nodes
    double o[length][param->n_class];       // output nodes
    
    double out_error[length][param->n_class];   // output error
    
    double delta_h[length][param->n_hidden];
    double delta_o[length][param->n_class];
    
    double tmp;
    
    int i,j,k;
    
    for(i=0; i<length; i++)
    {
        // Compute h
        // h = activation(x*wh)
        for(j=0; j<param->n_hidden; j++)
        {
            h[i][j] = 0;
            
            for(k=0; k<param->n_dimension; k++)
            {
                h[i][j] += x[i][k] * param->wh[k][j];
            }
            h[i][j] = 1 / (1 + exp(-h[i][j]));
            
            delta_h[i][j] = h[i][j] * (1-h[i][j]);  // Compute delta h here for optimization purpose
        }
        
        // Compute o
        // o = activation(h*wo)
        for(j=0; j<param->n_class; j++)
        {
            o[i][j] = 0;
            
            for(k=0; k<param->n_hidden; k++)
            {
                o[i][j] += h[i][k] * param->wo[k][j];
            }
            o[i][j] = 1 / (1 + exp(-o[i][j]));
            
            out_error[i][j] = -y[i][j] + o[i][j]; // Compute out error here for optimization purpose
            
            delta_o[i][j] = out_error[i][j] * o[i][j] * (1-o[i][j]);        // Compute delta o here for optimization purpose
        }
        
        // Compute rest of delta h
        for(j=0; j<param->n_hidden; j++)
        {
            tmp = 0;
            for(k=0; k<param->n_class; k++)
            {
                tmp += delta_o[i][k] * param->wo[j][k];
            }
            delta_h[i][j] *= tmp;
        }
    }
    
    // Update wo
    for(i=0; i<param->n_hidden; i++)
    {
        for(j=0; j<param->n_class; j++)
        {
            tmp = 0;
            for(k=0; k<length; k++)
            {
                tmp += h[k][i] * delta_o[k][j];
            }
            param->wo[i][j] -= param->alpha * tmp;
        }
    }
    
    // Update wh
    for(i=0; i<param->n_dimension; i++)
    {
        for(j=0; j<param->n_hidden; j++)
        {
            tmp = 0;
            for(k=0; k<length; k++)
            {
                tmp += x[k][i] * delta_h[k][j];
            }
            param->wh[i][j] -= param->alpha * tmp;
        }
    }
}

void jknn_classify(jknn_param *param, double* x, double* result)
{
    double h[param->n_hidden];      // hidden nodes
    
    int i,j;
    
    for(i=0; i<param->n_hidden; i++)
    {
        h[i] = 0;
        for(j=0; j<param->n_dimension; j++)
        {
            h[i] += x[j] * param->wh[j][i];
        }
        h[i] = 1 / (1 + exp(-h[i]));
    }
    
    for(i=0; i<param->n_class; i++)
    {
        result[i] = 0;
        for(j=0; j<param->n_hidden; j++)
        {
            result[i] += h[j] * param->wo[j][i];
        }
        result[i] = 1 / (1 + exp(-result[i]));
    }
}


void jknn_init_param(jknn_param *param, int _n_dimension, int _n_hidden, int _n_class, double _alpha)
{
    int i;
    
    param->n_dimension = _n_dimension;
    param->n_hidden = _n_hidden;
    param->n_class = _n_class;
    param->alpha = _alpha;
    
    param->wh = Malloc(double *, param->n_dimension);
    param->wo = Malloc(double *, param->n_hidden);
    
    double *_wh = Malloc(double, (param->n_dimension*param->n_hidden));
    double *_wo = Malloc(double, (param->n_hidden*param->n_class));
    
    for(i=0; i<(param->n_dimension*param->n_hidden); i++)
    {
        _wh[i] = (rand()%10001)*0.0001;
    }
    for(i=0; i<(param->n_hidden*param->n_class); i++)
    {
        _wo[i] = (rand()%10001)*0.0001;
    }
    
    for(i=0; i<param->n_dimension; i++)
    {
        param->wh[i] = &_wh[i*param->n_hidden];
    }
    for(i=0; i<param->n_hidden; i++)
    {
        param->wo[i] = &_wo[i*param->n_class];
    }
}

void jknn_free_param(jknn_param *param)
{
    free(param->wh[0]);
    free(param->wo[0]);
    
    free(param->wh);
    free(param->wo);
}
