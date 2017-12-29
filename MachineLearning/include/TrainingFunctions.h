#pragma once

#include "NeuralNet.h"

namespace UAFML
{
af::array InitializeWeights(dim_t num_elements, const af::dtype type = f32, double min = -0.1, double max = 0.1);
af::array RandomPermutation(dim_t number, dim_t max_value);

bool CheckGradient(NeuralNet &network, af::dim4 &input_size, dim_t num_classes, int checks = 1000, const af::dtype type = f32);

void MiniBatchGradientDescent(double alpha, double momentum, dim_t batch_size, NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights);
void StochasticGradientDescent(double alpha, double momentum, NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights);
void BatchGradientDescent(double alpha, NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights);
void ConjugateGradientDescent(NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights);

// HACKY - Improve or remove!
bool SingleExtendedGradientDescent(double &alpha, double &momentum, dim_t batch_size, unsigned int &iteration, unsigned int &epoch, af::array &tally, NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights);
}