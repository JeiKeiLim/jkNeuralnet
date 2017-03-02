//
//  JKNeuralNet.h
//  NeuralNet
//
//  Created by JeiKei on 2016. 12. 6..
//  Copyright © 2016년 JeiKei. All rights reserved.
//

#ifndef JKNeuralNet_h
#define JKNeuralNet_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef struct jknn_param {
    double alpha;
    int n_dimension;
    int n_hidden;
    int n_class;
    
    double **wh;
    double **wo;
} jknn_param;

double activation_sigmoid(double value);
double activation_relu(double value);

void jknn_init_param(jknn_param *param, int _n_dimension, int _n_hidden, int _n_class, double _alpha);
void jknn_free_param(jknn_param *param);
void jknn_classify(jknn_param *param, double* x, double* result);
void jknn_train(jknn_param *param, double** x, double** y, int data_length);
int jkGetArgMax(double *data, int n);
#endif /* JKNeuralNet_h */
