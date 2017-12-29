#include "pch.h"
#include "NeuralNet.h"

namespace UAFML
{

NeuralNet::NeuralNet(double lambda, double dropout) :
	_lambda(0.0),
	_dropout(0.0)
{
	SetLambda(lambda);
	SetDropout(dropout);
}

NeuralNet::~NeuralNet()
{
}

void NeuralNet::SetLambda(double value)
{
	_lambda = value;
}

void NeuralNet::SetDropout(double value)
{
	_dropout = value;
}

std::vector<af::dim4> NeuralNet::GetLayerSizes()
{
	std::vector<af::dim4> output;
	for (auto iter = _layers.begin(); iter != _layers.end(); ++iter)
	{
		output.emplace_back((*iter)->GetLayerSize());
	}
	return output;
}

dim_t NeuralNet::GetWeightsSize()
{
	dim_t elements = 0;
	for (auto iter = _layers.begin(); iter != _layers.end(); ++iter)
	{
		elements += (*iter)->GetLayerSize().elements();
	}
	return elements;
}

af::array NeuralNet::ForwardPropagate(const af::array &inputs, const af::array &weights)
{
	af::array current_values = inputs;
	af::array theta;
	std::pair<dim_t,dim_t> weight_range(-1,-1);
	af::dim4 layer_size;
	dim_t elements;
	for (auto iter = _layers.begin(); iter != _layers.end(); ++iter)
	{
		layer_size = (*iter)->GetLayerSize();
		elements = layer_size.elements();
		if(0 < elements)
		{
			weight_range.first = weight_range.second + 1;
			weight_range.second += elements;
			theta = af::moddims(weights(af::seq((double)weight_range.first, (double)weight_range.second)), layer_size);
		}

		if (!(*iter)->ForwardPropagate(current_values, theta))
		{
			std::cout << "ForwardPropagate failed!\n";
			throw std::exception("ForwardPropagate failed!");
		}
	}
	return current_values;
}

af::array NeuralNet::BackPropagate(const af::array &output, const af::array &truth, const af::array &weights)
{
	af::array gradient(weights.dims());
	af::array gradient_chunk;
	af::array current_values = output;
	af::array theta;
	af::dim4 layer_size;
	dim_t elements = weights.elements();
	std::pair<dim_t, dim_t> weight_range(elements, elements);
	for (auto iter = _layers.rbegin(); iter != _layers.rend(); ++iter)
	{
		layer_size = (*iter)->GetLayerSize();
		elements = layer_size.elements();
		if (0 < elements)
		{
			weight_range.second = weight_range.first - 1;
			weight_range.first -= elements;
			theta = af::moddims(weights(af::seq((double)weight_range.first, (double)weight_range.second)), layer_size);
		}

		if (_layers.rbegin() == iter)
		{
			gradient_chunk = (*iter)->CalculateError(current_values, truth, theta);
		}
		else
		{
			gradient_chunk = (*iter)->BackPropagate(current_values, theta);
		}
		if (0 < elements)
		{
			gradient(af::seq((double)weight_range.first, (double)weight_range.second)) = af::flat(gradient_chunk);
		}
	}
	return gradient + weights * _lambda / weights.elements();
}

double NeuralNet::CalculateCost(const af::array &output, const af::array &truth, const af::array &weights)
{
	if (0 == _layers.size())
		return NAN;

	auto iter = _layers.rbegin();
	double cost = (*iter)->CalculateCost(output, truth) + af::sum<double>(af::pow(weights,2)) * _lambda / (2 * weights.elements());
	return cost;
}

}