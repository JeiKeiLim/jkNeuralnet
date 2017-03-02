# jkNeuralnet

Implementation of Neural network using Logistic function.

##Parameter structure description
<B>jknn_param</B> structure offers information for neural networ. Here is some description for the elements.
- alpha : learning rate
- n_dimension : input dimension, so called feature number or dimension
- n_hidden : number of hidden node
- n_class : number of class
- wh : weights for hidden nodes. Its dimension will be [<B>n_dimension</B>]X[<B>n_hidden</B>]
- wo : weights for output nodes. Its dimension will be [<B>n_hidden</B>]X[<B>n_class</B>]


##Function description
void <B>jknn_init_param</B>(jknn_param <B>*param</B>, int <B>_n_dimension</B>, int <B>_n_hidden</B>, int <B>_n_class</B>, double <B>_alpha</B>);
- Initialise parameters for neural network.
- It creates memory spaces for wh and wo automatically which requires to run <B>jknn_free_param</B> later to free allocated memory
- All wh and wo parameters are initialised with random numbers.

void <B>jknn_free_param</B>(jknn_param <B>*param</B>);
- Free memory spaces

void <B>jknn_train</B>(jknn_param <B>*param</B>, double <B>**x</B>, char <B>**y</B>, int <B>data_length</B>);
- Train parameters for neural network.
- Its learning rate is already defined in <B>param->alpha</B> you can change it as you wish
- <B>**x</B> : Training data. It has to have same dimension with [<B>n_dimension</B>]X[<B>data_length</B>]
- <B>**y</B> : Training label. It has to have same dimension with [<B>n_class</B>]X[<B>data_length</B>]


void <B>jknn_classify</B>(jknn_param <B>*param</B>, double <B>*x</B>, double <B>*result</B>);
- Classify with learned network
- <B>*x</B> : Testing data. It has to have same dimension with [<B>n_dimension</B>]
- <B>*result</B> : Return result. It returns probability of classes.
- It does not support argmax yet




# Usage example for XOR
```C
jknn_param param;
jknn_init_param(&param, 2, 4, 2, 0.7);

int i;

double **x = (double **)malloc(sizeof(double *)*4);
double **y = (double **)malloc(sizeof(double *)*4);

for(i=0; i<4; i++)
{
  x[i] = (double *)malloc(sizeof(double)*4);
  y[i] = (double *)malloc(sizeof(double)*2);
}

x[0][0] = 0;        x[0][1] = 0;
x[1][0] = 0;        x[1][1] = 1;
x[2][0] = 1;        x[2][1] = 0;
x[3][0] = 1;        x[3][1] = 1;

y[0][0] = 1;        y[0][1] = 0;
y[1][0] = 0;        y[1][1] = 1;
y[2][0] = 0;        y[2][1] = 1;
y[3][0] = 1;        y[3][1] = 0;

for(i=0; i<1000; i++)
  jknn_train(&param, x, y, 4);

double *result = Malloc(double, 2);
for(i=0; i<4; i++)
{
  jknn_classify(&param, x[i], result);
  printf("0 : %f,\t 1 : %f\n", result[0], result[1]);
}

for(i=0; i<4; i++)
{
  free(x[i]);
  free(y[i]);
}
free(x);
free(y);
free(result);

jknn_free_param(&param);
```
