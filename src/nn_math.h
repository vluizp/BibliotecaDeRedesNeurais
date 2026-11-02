// nn_math.h
#ifndef NN_MATH_H
#define NN_MATH_H

#include "nn_types.h"

float sigmoide(float x);
float derivada_sigmoide(float x);
float calcula_erro(struct network *net);

#endif // NN_MATH_H